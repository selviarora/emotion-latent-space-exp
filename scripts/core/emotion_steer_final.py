#!/usr/bin/env python3
"""
Emotion Steering with FINAL Conv1d Mapper + HiFi-GAN
This should actually work!
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import argparse
import sys
import json
from librosa.filters import mel as librosa_mel_fn

sys.path.insert(0, 'models/hifigan_repo')
from models import Generator

# Settings
HIFIGAN_SR = 22050
WAV2VEC_SR = 16000
N_MELS = 80
N_FFT = 1024
HOP_SIZE = 256
WIN_SIZE = 1024

# Final mapper architecture (must match training)
class ConvBlock(nn.Module):
    def __init__(self, ch, k=7, d=1):
        super().__init__()
        pad = (k-1)//2 * d
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, dilation=d, padding=pad)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(ch)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + y

class Wav2VecToMelTemporal(nn.Module):
    def __init__(self, in_ch=768, hidden=512, n_mels=80):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.Sequential(
            ConvBlock(hidden, k=7, d=1),
            ConvBlock(hidden, k=7, d=2),
            ConvBlock(hidden, k=7, d=4),
            ConvBlock(hidden, k=7, d=8),   # upgraded architecture
            ConvBlock(hidden, k=7, d=16),  # upgraded architecture
        )
        self.proj = nn.Conv1d(hidden, n_mels, 1)

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.inp(X)
        H = self.blocks(H)
        M = self.proj(H)
        return M.transpose(1, 2)

# HiFi-GAN mel
mel_basis = {}
hann_window = {}

def mel_spectrogram_hifigan(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec

def load_hifigan_vocoder(checkpoint_path, device):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    with open('models/hifigan_repo/config_v1.json') as f:
        config = AttrDict(json.load(f))
    
    generator = Generator(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    return generator, config

def extract_wav2vec_features(audio, sr):
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
    
    audio = audio.astype(np.float32)
    inp = proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        out = model(**inp)
        features = out.hidden_states[5].squeeze(0)
    
    return features

def main():
    ap = argparse.ArgumentParser(description="Emotion Steering - FINAL VERSION")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--slider", type=float, default=0.0, help="-1 (calm) to +1 (angry)")
    ap.add_argument("--mode", choices=["subtle", "creative"], default="subtle")
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load HiFi-GAN (fine-tuned on mapper mels)
    print("Loading HiFi-GAN vocoder...")
    vocoder, config = load_hifigan_vocoder('models/hifigan_finetuned.pt', device)
    
    # Load emotion axis
    if args.mode == "subtle":
        axis = np.load('models/emotion_axis_layer5.npy')
        print("Mode: SUBTLE (Layer5)")
    else:
        axis = np.load('models/emotion_axis_hybrid.npy')
        print("Mode: CREATIVE (Hybrid)")
    
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    axis = torch.FloatTensor(axis).to(device)
    
    # Load audio ONCE, trim, then resample both branches identically
    print(f"Loading: {args.input}")
    audio_orig, sr_orig = librosa.load(args.input, sr=None)
    # NO TRIMMING - must match training (perfect alignment)
    
    audio_w2v = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=WAV2VEC_SR)
    audio_hifi = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=HIFIGAN_SR)
    
    # RMS normalization (same as training)
    def rms_norm(x, target=0.1, eps=1e-8):
        r = np.sqrt((x**2).mean() + eps)
        return x * (target / (r + eps))
    
    audio_w2v = rms_norm(audio_w2v)
    audio_hifi = rms_norm(audio_hifi)
    
    # Extract GT mel FIRST to get EXACT frame count (don't estimate!)
    audio_hifi_tensor = torch.FloatTensor(audio_hifi).unsqueeze(0).to(device)
    mel_gt = mel_spectrogram_hifigan(audio_hifi_tensor, N_FFT, N_MELS, HIFIGAN_SR, HOP_SIZE, WIN_SIZE, 0, 8000, center=False)
    T_mel_actual = mel_gt.shape[2]  # Get ACTUAL frame count from GT mel
    print(f"Actual GT mel frames: {T_mel_actual}")
    
    # Extract wav2vec features
    print("Extracting wav2vec features...")
    wav2vec_feats = extract_wav2vec_features(audio_w2v, WAV2VEC_SR).to(device)
    print(f"Wav2vec frames: {wav2vec_feats.shape[0]}")
    
    # Get current score
    with torch.no_grad():
        energy = wav2vec_feats.pow(2).sum(dim=1)
        weights = energy / (energy.sum() + 1e-8)
        emb_current = (wav2vec_feats * weights.unsqueeze(1)).sum(dim=0)
        current_score = float(emb_current @ axis)
    
    print(f"Current projection: {current_score:.3f}")
    print(f"Target slider: {args.slider:.2f}")
    
    # Apply steering
    delta = args.slider * args.strength
    print(f"Steering delta: {delta:+.2f}")
    
    wav2vec_steered = wav2vec_feats + (delta * axis.unsqueeze(0))
    
    # Upsample to match EXACT GT mel length (not calculated estimate)
    print(f"Upsampling wav2vec {wav2vec_steered.shape[0]} → {T_mel_actual} frames (GT mel length)...")
    wav2vec_steered_up = torch.nn.functional.interpolate(
        wav2vec_steered.T.unsqueeze(0),
        size=T_mel_actual,
        mode='linear',
        align_corners=False
    ).squeeze(0).T
    
    print(f"✓ Interpolation check: wav2vec_up={wav2vec_steered_up.shape[0]}, T_mel_actual={T_mel_actual}, match={wav2vec_steered_up.shape[0] == T_mel_actual}")
    
    # Load FINAL mapper
    print("Loading FINAL mapper...")
    mapper = Wav2VecToMelTemporal(in_ch=768, hidden=512, n_mels=N_MELS).to(device)
    mapper.load_state_dict(torch.load('models/wav2vec_to_mel_mapper_final.pt', map_location=device))
    mapper.eval()
    
    # Map to mel
    print("Mapping to mel...")
    with torch.no_grad():
        mel_pred = mapper(wav2vec_steered_up.unsqueeze(0))  # [1, T_mel, 80]
        mel_pred = mel_pred.squeeze(0).T.unsqueeze(0)  # [1, 80, T_mel]
    
    # Print BEFORE clamping
    print(f"📊 Mel BEFORE clamp: min={mel_pred.min():.3f}, mean={mel_pred.mean():.3f}, max={mel_pred.max():.3f}")
    
    # Clamp to safe range
    mel_pred_clamped = torch.clamp(mel_pred, min=-11.5, max=2.5)
    
    # Print AFTER clamping
    print(f"📊 Mel AFTER clamp:  min={mel_pred_clamped.min():.3f}, mean={mel_pred_clamped.mean():.3f}, max={mel_pred_clamped.max():.3f}")
    print(f"   Expected range: [-11 to 2.5], std={mel_pred_clamped.std():.3f}")
    
    mel_pred = mel_pred_clamped
    
    # Compare to GT (already extracted earlier)
    print(f"📊 Mel GT:   shape={mel_gt.shape}, range=[{mel_gt.min():.2f}, {mel_gt.max():.2f}], mean={mel_gt.mean():.2f}, std={mel_gt.std():.2f}")
    
    # Final frame count check
    print(f"\n✓ Frame count check before vocoder:")
    print(f"  mel_pred shape: {mel_pred.shape} [should be 1, 80, T]")
    print(f"  mel_gt shape:   {mel_gt.shape} [should be 1, 80, T]")
    print(f"  Frame match: {mel_pred.shape[2] == mel_gt.shape[2]} (off by {abs(mel_pred.shape[2] - mel_gt.shape[2])} frames)")
    
    # Vocoder
    print("\nVocoding with HiFi-GAN...")
    with torch.no_grad():
        audio_out = vocoder(mel_pred).squeeze().cpu().numpy()
    
    # Resample to 16kHz
    audio_out = librosa.resample(audio_out, orig_sr=HIFIGAN_SR, target_sr=16000)
    audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8) * 0.95
    
    # Save
    sf.write(args.output, audio_out, 16000)
    print(f"\n✓ Saved: {args.output}")
    
    print("\n" + "="*60)
    print("🎧 MOMENT OF TRUTH - LISTEN TO THE OUTPUT!")
    print(f"Expected: {'CALMER' if args.slider < 0 else 'MORE INTENSE'}")
    print("="*60)

if __name__ == "__main__":
    main()

