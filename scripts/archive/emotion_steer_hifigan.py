#!/usr/bin/env python3
"""
Emotion Steering with HiFi-GAN Vocoder
The real deal - proper vocoder with matching mel format
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import argparse
import sys
import os

# Add HiFi-GAN repo to path
sys.path.insert(0, 'models/hifigan_repo')
from models import Generator
from meldataset import mel_spectrogram as mel_spec_hifigan
import json

# HiFi-GAN settings
HIFIGAN_SR = 22050
WAV2VEC_SR = 16000
N_MELS = 80
N_FFT = 1024
HOP_SIZE = 256

class Wav2VecToMelMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N_MELS)
        )
    
    def forward(self, x):
        return self.layers(x)

def load_hifigan_vocoder(checkpoint_path, device):
    """Load pretrained HiFi-GAN vocoder"""
    # Load config
    config_path = 'models/hifigan_repo/config_v1.json'
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Convert dict to AttrDict (HiFi-GAN expects attributes)
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    config = AttrDict(config_dict)
    
    # Create generator
    generator = Generator(config).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    return generator, config

def extract_wav2vec_features(audio, sr):
    """Extract wav2vec layer5 features"""
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
    
    audio = audio.astype(np.float32)
    inp = proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        out = model(**inp)
        features = out.hidden_states[5].squeeze(0)
    
    return features

def main():
    ap = argparse.ArgumentParser(description="Emotion Steering with HiFi-GAN")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--slider", type=float, default=0.0, help="-1 (calm) to +1 (angry)")
    ap.add_argument("--mode", choices=["subtle", "creative"], default="subtle")
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load HiFi-GAN vocoder
    print("Loading HiFi-GAN vocoder...")
    vocoder, config = load_hifigan_vocoder('models/hifigan/g_02500000', device)
    
    # Load emotion axis (RAW)
    if args.mode == "subtle":
        axis = np.load('models/emotion_axis_layer5.npy')
        print("Mode: SUBTLE (Layer5)")
    else:
        axis = np.load('models/emotion_axis_hybrid.npy')
        print("Mode: CREATIVE (Hybrid)")
    
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    axis = torch.FloatTensor(axis).to(device)
    
    # Load audio at BOTH sample rates
    print(f"Loading: {args.input}")
    audio_w2v, _ = librosa.load(args.input, sr=WAV2VEC_SR)
    audio_w2v, _ = librosa.effects.trim(audio_w2v, top_db=20)
    
    # Also load at HiFi-GAN SR to compute expected mel length
    audio_hifi, _ = librosa.load(args.input, sr=HIFIGAN_SR)
    N = len(audio_hifi)
    
    # Compute expected mel length (matches training, center=False)
    import math
    T_mel = max(1, 1 + (N - N_FFT) // HOP_SIZE)
    print(f"Expected mel frames: {T_mel}")
    
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
    
    # FIX 1: Upsample wav2vec to match expected mel length (same as training)
    print(f"Upsampling wav2vec {wav2vec_steered.shape[0]} → {T_mel} frames...")
    wav2vec_steered_up = torch.nn.functional.interpolate(
        wav2vec_steered.T.unsqueeze(0),  # [1, 768, T_w2v]
        size=T_mel,
        mode='linear',
        align_corners=False
    ).squeeze(0).T  # [T_mel, 768]
    
    print(f"Upsampled shape: {wav2vec_steered_up.shape}")
    
    # Load mapper
    print("Loading mapper...")
    mapper = Wav2VecToMelMapper().to(device)
    mapper.load_state_dict(torch.load('models/wav2vec_to_mel_mapper_hifigan.pt', map_location=device))
    mapper.eval()
    
    # Map to mel
    print("Mapping to mel...")
    with torch.no_grad():
        mel_pred = mapper(wav2vec_steered_up)  # [T_mel, 80]
        mel_pred = mel_pred.T.unsqueeze(0)  # [1, 80, T_mel]
    
    # FIX 2: Clamp mel to safe log-mel range for HiFi-GAN
    mel_pred = torch.clamp(mel_pred, min=-11.5, max=2.5)
    print(f"Mel PRED shape: {mel_pred.shape}, range: [{mel_pred.min():.2f}, {mel_pred.max():.2f}], mean: {mel_pred.mean():.2f}, std: {mel_pred.std():.2f}")
    
    # Compare to GT mel
    audio_hifi_tensor = torch.FloatTensor(audio_hifi).unsqueeze(0)
    mel_gt = mel_spec_hifigan(audio_hifi_tensor, 1024, 80, HIFIGAN_SR, HOP_SIZE, 1024, 0, 8000, center=False)
    print(f"Mel GT   shape: {mel_gt.shape}, range: [{mel_gt.min():.2f}, {mel_gt.max():.2f}], mean: {mel_gt.mean():.2f}, std: {mel_gt.std():.2f}")
    print(f"⚠️ If PRED std << GT std → over-smoothed (needs Conv1d!)")
    
    # Vocoder: mel → audio
    print("Vocoding with HiFi-GAN...")
    with torch.no_grad():
        audio_out = vocoder(mel_pred).squeeze().cpu().numpy()
    
    # Resample to 16kHz for consistency
    audio_out = librosa.resample(audio_out, orig_sr=HIFIGAN_SR, target_sr=16000)
    
    # Normalize
    audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8) * 0.95
    
    # Save
    sf.write(args.output, audio_out, 16000)
    print(f"\n✓ Saved: {args.output}")
    
    # Verify
    print("\nVerifying...")
    audio_out_w2v, _ = librosa.load(args.output, sr=WAV2VEC_SR)
    wav2vec_new = extract_wav2vec_features(audio_out_w2v, WAV2VEC_SR).to(device)
    
    with torch.no_grad():
        energy_new = wav2vec_new.pow(2).sum(dim=1)
        weights_new = energy_new / (energy_new.sum() + 1e-8)
        emb_new = (wav2vec_new * weights_new.unsqueeze(1)).sum(dim=0)
        new_score = float(emb_new @ axis)
    
    print(f"New projection: {new_score:.3f}")
    print(f"Change: {new_score - current_score:+.3f}")
    
    print("\n" + "="*60)
    print("🎧 LISTEN TO THE OUTPUT!")
    print(f"Does it sound {'CALMER' if args.slider < 0 else 'MORE INTENSE'}?")
    print("="*60)

if __name__ == "__main__":
    main()

