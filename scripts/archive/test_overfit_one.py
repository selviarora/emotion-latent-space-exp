#!/usr/bin/env python3
"""
One-file overfit test - FAST diagnostic
If it can't overfit ONE file, alignment/scale is broken
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from librosa.filters import mel as librosa_mel_fn
import sys
sys.path.insert(0, 'models/hifigan_repo')
from models import Generator
import json

HIFIGAN_SR = 22050
WAV2VEC_SR = 16000
N_FFT = 1024
N_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000

# Mapper architecture
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
    
    return generator

# Load one file
audio_path = "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01/03-01-05-02-02-01-01.wav"

# Load ONCE, trim, then resample both branches
audio_orig, sr_orig = librosa.load(audio_path, sr=None)
# NO TRIMMING - ensures perfect temporal alignment

wav_hifi = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=HIFIGAN_SR).astype(np.float32)
wav_w2v = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=WAV2VEC_SR).astype(np.float32)

# RMS normalization
def rms_norm(x, target=0.1, eps=1e-8):
    r = np.sqrt((x**2).mean() + eps)
    return x * (target / (r + eps))

wav_hifi = rms_norm(wav_hifi)
wav_w2v = rms_norm(wav_w2v)

# Extract wav2vec features
proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

inp = proc(wav_w2v, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model_w2v(**inp)
    wav2vec_feats = out.hidden_states[5].squeeze(0)  # [T_w2v, 768]

# Extract GT mel to get EXACT frame count
wav_hifi_tensor = torch.FloatTensor(wav_hifi).unsqueeze(0)
mel_gt = mel_spectrogram_hifigan(wav_hifi_tensor, N_FFT, N_MELS, HIFIGAN_SR, HOP_SIZE, WIN_SIZE, FMIN, FMAX, center=False)
mel_gt = mel_gt.squeeze(0).T  # [T_mel_actual, 80]

# Use ACTUAL mel frame count
T_mel_actual = mel_gt.shape[0]
T_w2v = wav2vec_feats.shape[0]

# Always upsample to match EXACT GT mel length
if T_w2v != T_mel_actual:
    wav2vec_feats = torch.nn.functional.interpolate(
        wav2vec_feats.T.unsqueeze(0),
        size=T_mel_actual,
        mode='linear',
        align_corners=False
    ).squeeze(0).T

print(f"Wav2Vec shape: {wav2vec_feats.shape}")
print(f"Mel GT shape: {mel_gt.shape}")
print(f"T matches: {wav2vec_feats.shape[0] == mel_gt.shape[0]}")

# Train mapper to overfit this ONE file
device = torch.device('cpu')
mapper = Wav2VecToMelTemporal().to(device)
optimizer = optim.Adam(mapper.parameters(), lr=1e-3)

wav2vec_batch = wav2vec_feats.unsqueeze(0).to(device)  # [1, T, 768]
mel_batch = mel_gt.unsqueeze(0).to(device)  # [1, T, 80]

print("\n🔥 Overfitting on ONE file (500 steps)...")
for step in range(500):
    optimizer.zero_grad()
    mel_pred = mapper(wav2vec_batch)
    loss = nn.functional.l1_loss(mel_pred, mel_batch)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}: Loss={loss.item():.4f}")

# Final check
with torch.no_grad():
    mel_pred = mapper(wav2vec_batch).squeeze(0)
    
print(f"\nFinal:")
print(f"  pred mean/std/min/max: {mel_pred.mean():.3f} / {mel_pred.std():.3f} / {mel_pred.min():.3f} / {mel_pred.max():.3f}")
print(f"  gt   mean/std/min/max: {mel_gt.mean():.3f} / {mel_gt.std():.3f} / {mel_gt.min():.3f} / {mel_gt.max():.3f}")

# Vocode it
vocoder = load_hifigan_vocoder('models/hifigan/g_02500000', device)
mel_pred_vocode = torch.clamp(mel_pred.T.unsqueeze(0), min=-11.5, max=2.5)

with torch.no_grad():
    audio_out = vocoder(mel_pred_vocode).squeeze().cpu().numpy()

audio_out = librosa.resample(audio_out, orig_sr=HIFIGAN_SR, target_sr=16000)
audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8) * 0.95

sf.write('overfit_test.wav', audio_out, 16000)
print("\n✓ Saved: overfit_test.wav")
print("\n🎧 LISTEN: If this sounds clean, alignment is fixed!")
print("If still windy → alignment/scale still broken")

