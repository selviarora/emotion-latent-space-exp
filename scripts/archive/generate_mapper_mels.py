#!/usr/bin/env python3
"""
Generate predicted mels from the trained mapper for HiFi-GAN fine-tuning.
"""
import torch
import torch.nn as nn
import numpy as np
import librosa
import glob
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

WAV2VEC_SR = 16000
HIFIGAN_SR = 22050
N_FFT = 1024
N_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024

# Model classes
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
            ConvBlock(hidden, k=7, d=8),
            ConvBlock(hidden, k=7, d=16),
        )
        self.proj = nn.Conv1d(hidden, n_mels, 1)
    
    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.inp(X)
        H = self.blocks(H)
        M = self.proj(H)
        return M.transpose(1, 2)

def rms_norm(x, target=0.1, eps=1e-8):
    r = np.sqrt((x**2).mean() + eps)
    return x * (target / (r + eps))

# HiFi-GAN mel extraction
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def mel_spectrogram_hifigan(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Actor_* dirs")
    ap.add_argument("--out_dir", default="mapper_mels", help="Output directory for predicted mels")
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")
    
    # Load mapper
    print("Loading mapper...")
    mapper = Wav2VecToMelTemporal(in_ch=768, hidden=512, n_mels=N_MELS).to(device)
    mapper.load_state_dict(torch.load('models/wav2vec_to_mel_mapper_final.pt', map_location=device))
    mapper.eval()
    
    # Load wav2vec model
    print("Loading Wav2Vec2...")
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).to(device).eval()
    
    # Get all audio files
    audio_paths = sorted(glob.glob(f"{args.data}/Actor_*/03-01-*-02-02-01-*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    # Process each file
    for audio_path in tqdm(audio_paths, desc="Generating mels"):
        # Load audio
        audio_orig, sr_orig = librosa.load(audio_path, sr=None)
        
        # NO trimming (same as training)
        audio_w2v = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=WAV2VEC_SR)
        audio_hifi = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=HIFIGAN_SR)
        
        # RMS norm (same as training)
        audio_w2v = rms_norm(audio_w2v)
        audio_hifi = rms_norm(audio_hifi)
        
        # Extract GT mel (to get exact frame count)
        audio_hifi_tensor = torch.FloatTensor(audio_hifi).unsqueeze(0).to(device)
        mel_gt = mel_spectrogram_hifigan(audio_hifi_tensor, N_FFT, N_MELS, HIFIGAN_SR, HOP_SIZE, WIN_SIZE, 0, 8000, center=False)
        T_mel = mel_gt.shape[2]
        
        # Extract wav2vec features
        with torch.no_grad():
            inp = proc(audio_w2v, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model_w2v(**inp)
            wav2vec_feats = out.hidden_states[5].squeeze(0)
        
        # Upsample to match mel length
        wav2vec_up = torch.nn.functional.interpolate(
            wav2vec_feats.T.unsqueeze(0),
            size=T_mel,
            mode='linear',
            align_corners=False
        ).squeeze(0).T
        
        # Predict mel
        with torch.no_grad():
            mel_pred = mapper(wav2vec_up.unsqueeze(0))  # [1, T, 80]
            mel_pred = mel_pred.squeeze(0).T  # [80, T]
        
        # Save predicted mel and corresponding audio
        basename = os.path.basename(audio_path).replace('.wav', '')
        
        # Save predicted mel
        mel_path = os.path.join(args.out_dir, f"{basename}_mel.npy")
        np.save(mel_path, mel_pred.cpu().numpy())
        
        # Save corresponding audio (for pairing during fine-tuning)
        audio_path_out = os.path.join(args.out_dir, f"{basename}.wav")
        import soundfile as sf
        sf.write(audio_path_out, audio_hifi, HIFIGAN_SR)
    
    print(f"\n✅ Generated {len(audio_paths)} predicted mels in {args.out_dir}/")
    print(f"   Each file has: <name>_mel.npy (predicted mel) and <name>.wav (audio)")

if __name__ == "__main__":
    main()

