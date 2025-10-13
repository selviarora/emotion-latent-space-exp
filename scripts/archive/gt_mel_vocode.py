#!/usr/bin/env python3
import torch, librosa, soundfile as sf
import sys
sys.path.insert(0, 'models/hifigan_repo')
from meldataset import mel_spectrogram as mel_spec_hifigan
from models import Generator
import json

HIFIGAN_SR=22050; N_FFT=1024; N_MELS=80; HOP=256; WIN=1024; FMIN=0; FMAX=8000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load vocoder (same as your script)
cfg = json.load(open('models/hifigan_repo/config_v1.json'))
class AttrDict(dict): 
    __getattr__=dict.get
    __setattr__=dict.__setitem__

g = Generator(AttrDict(cfg)).to(device)
state = torch.load('models/hifigan/g_02500000', map_location=device)
g.load_state_dict(state['generator'])
g.eval()
g.remove_weight_norm()

# WAV -> GT mel (exact HiFi-GAN settings)
y, _ = librosa.load("/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01/03-01-05-02-02-01-01.wav", sr=HIFIGAN_SR)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

# Fix for librosa API
from librosa.filters import mel as librosa_mel_fn

mel_basis_gt = {}
hann_window_gt = {}

def mel_spectrogram_fixed(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis_gt, hann_window_gt
    if fmax not in mel_basis_gt:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis_gt[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window_gt[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window_gt[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(mel_basis_gt[str(fmax)+'_'+str(y.device)], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec

mel = mel_spectrogram_fixed(y, N_FFT, N_MELS, HIFIGAN_SR, HOP, WIN, FMIN, FMAX, center=False)

print(f"GT Mel: shape={mel.shape}, min={mel.min():.2f}, max={mel.max():.2f}, mean={mel.mean():.2f}, std={mel.std():.2f}")

with torch.no_grad():
    out = g(mel).squeeze().cpu().numpy()

sf.write("gt_vocode.wav", out, HIFIGAN_SR)
print("✓ Wrote gt_vocode.wav - this should sound CLEAN!")


