#!/usr/bin/env python3
"""
Round-trip test: audio → GT mel → HiFi-GAN → audio
This proves the vocoder/config is correct
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import sys
import json

sys.path.insert(0, 'models/hifigan_repo')
from models import Generator
from meldataset import mel_spectrogram as mel_spec_hifigan

HIFIGAN_SR = 22050
N_FFT = 1024
HOP_SIZE = 256
WIN_SIZE = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000

# Load HiFi-GAN
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open('models/hifigan_repo/config_v1.json') as f:
    config = AttrDict(json.load(f))

device = torch.device('cpu')
vocoder = Generator(config).to(device)
checkpoint = torch.load('models/hifigan/g_02500000', map_location=device)
vocoder.load_state_dict(checkpoint['generator'])
vocoder.eval()
vocoder.remove_weight_norm()

# Load audio
input_path = "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01/03-01-05-02-02-01-01.wav"
audio, _ = librosa.load(input_path, sr=HIFIGAN_SR)
audio = torch.FloatTensor(audio).unsqueeze(0)

# Extract GT mel using HiFi-GAN's function
# Note: need to use the modified version from training script with keyword args
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}

def mel_spectrogram_fixed(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
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

mel_gt = mel_spectrogram_fixed(audio, N_FFT, N_MELS, HIFIGAN_SR, HOP_SIZE, WIN_SIZE, FMIN, FMAX, center=False)

# Print stats
print("GT Mel Stats:")
print(f"  Shape: {mel_gt.shape}")
print(f"  Min: {mel_gt.min():.2f}")
print(f"  Max: {mel_gt.max():.2f}")
print(f"  Mean: {mel_gt.mean():.2f}")
print(f"  Std: {mel_gt.std():.2f}")

# Vocode GT mel
with torch.no_grad():
    audio_out = vocoder(mel_gt).squeeze().cpu().numpy()

# Resample to 16k
audio_out = librosa.resample(audio_out, orig_sr=HIFIGAN_SR, target_sr=16000)
audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8) * 0.95

# Save
sf.write('test_roundtrip_gt.wav', audio_out, 16000)
print("\n✓ Saved: test_roundtrip_gt.wav")
print("This should sound CLEAN and IDENTICAL to original!")
print("If it's windy → vocoder/config mismatch")
print("If it's clean → mapper is the problem")

