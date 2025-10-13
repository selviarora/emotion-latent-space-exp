#!/usr/bin/env python3
"""
Griffin-Lim test: Can our mapper's mels produce intelligible speech?
"""
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Copy minimal requirements
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.core.emotion_steer_final import (
    Wav2VecToMelTemporal, ConvBlock, mel_spectrogram_hifigan,
    WAV2VEC_SR, HIFIGAN_SR, N_FFT, N_MELS, HOP_SIZE, WIN_SIZE
)

def rms_norm(x, target=0.1, eps=1e-8):
    r = np.sqrt((x**2).mean() + eps)
    return x * (target / (r + eps))

# Griffin-Lim
def griffin_lim(mel_db, n_iter=60, hop_length=256, win_length=1024, n_fft=1024):
    """Convert log-mel to audio using Griffin-Lim."""
    # Convert log-mel back to linear (mel_db is already log-compressed)
    mel_linear = np.exp(mel_db)
    
    # Inverse mel filterbank using pseudo-inverse
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=80, fmin=0, fmax=8000)
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    stft_magnitude = np.dot(mel_basis_pinv, mel_linear)
    
    # Ensure non-negative
    stft_magnitude = np.maximum(stft_magnitude, 0)
    
    # Griffin-Lim expects power spectrogram
    stft_power = stft_magnitude ** 2
    
    # Griffin-Lim
    audio = librosa.griffinlim(
        stft_power, 
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        center=False
    )
    
    return audio

# Load mapper
device = torch.device('cpu')
print("Loading mapper...")
mapper = Wav2VecToMelTemporal(in_ch=768, hidden=512, n_mels=N_MELS).to(device)
mapper.load_state_dict(torch.load('models/wav2vec_to_mel_mapper_final.pt', map_location=device))
mapper.eval()

# Load wav2vec
print("Loading Wav2Vec2...")
proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

# Test file
audio_path = "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01/03-01-05-02-02-01-01.wav"
print(f"Testing: {audio_path}")

# Load audio
audio_orig, sr_orig = librosa.load(audio_path, sr=None)
audio_w2v = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=WAV2VEC_SR)
audio_hifi = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=HIFIGAN_SR)

# RMS norm
audio_w2v = rms_norm(audio_w2v)
audio_hifi = rms_norm(audio_hifi)

# Extract GT mel
audio_hifi_tensor = torch.FloatTensor(audio_hifi).unsqueeze(0)
mel_gt = mel_spectrogram_hifigan(audio_hifi_tensor, N_FFT, N_MELS, HIFIGAN_SR, HOP_SIZE, WIN_SIZE, 0, 8000, center=False)
T_mel = mel_gt.shape[2]

# Extract wav2vec
inp = proc(audio_w2v, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model_w2v(**inp)
    wav2vec_feats = out.hidden_states[5].squeeze(0)

# Upsample
wav2vec_up = torch.nn.functional.interpolate(
    wav2vec_feats.T.unsqueeze(0),
    size=T_mel,
    mode='linear',
    align_corners=False
).squeeze(0).T

# Predict mel
with torch.no_grad():
    mel_pred = mapper(wav2vec_up.unsqueeze(0))
    mel_pred = mel_pred.squeeze(0).T.numpy()  # [80, T]

print(f"\n📊 Predicted mel shape: {mel_pred.shape}")
print(f"   Range: [{mel_pred.min():.2f}, {mel_pred.max():.2f}]")

# Griffin-Lim reconstruction
print("\n🔊 Running Griffin-Lim (32 iterations)...")
audio_gl = griffin_lim(mel_pred, n_iter=32, hop_length=HOP_SIZE, win_length=WIN_SIZE, n_fft=N_FFT)

# Save
output_path = "test_griffin_lim.wav"
sf.write(output_path, audio_gl, HIFIGAN_SR)
print(f"✅ Saved: {output_path}")

print("\n" + "="*60)
print("DIAGNOSTIC:")
print("="*60)
print("Listen to test_griffin_lim.wav:")
print("  • If INTELLIGIBLE (words clear, even if buzzy)")
print("    → Mapper mels are GOOD!")
print("    → Full HiFi-GAN fine-tuning will fix it!")
print()
print("  • If ROBOTIC/UNINTELLIGIBLE")
print("    → Mapper mels lack harmonic detail")
print("    → Need to improve mapper architecture")
print("="*60)

