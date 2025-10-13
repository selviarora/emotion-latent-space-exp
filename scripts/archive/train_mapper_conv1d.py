#!/usr/bin/env python3
"""
Train Wav2Vec → Mel Mapper WITH TEMPORAL CONTEXT (Conv1d)
This fixes the over-smoothing problem!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import glob
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm
import os
from librosa.filters import mel as librosa_mel_fn

# HiFi-GAN exact settings
HIFIGAN_SR = 22050
N_FFT = 1024
N_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000

# Wav2Vec settings
WAV2VEC_SR = 16000

# HiFi-GAN's exact mel extraction
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram_hifigan(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """HiFi-GAN's exact mel function"""
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
    spec = dynamic_range_compression_torch(spec)  # Log compression

    return spec

# NEW: Conv1d mapper with temporal context
class Wav2VecToMelMapperConv(nn.Module):
    def __init__(self, d_in=768, d_h=256, d_hidden=192, n_mels=80):
        super().__init__()
        # Project to smaller dim
        self.inp = nn.Linear(d_in, d_h)  # [T,768] → [T,256]
        
        # Temporal convolutions (THIS IS THE KEY FIX!)
        self.conv = nn.Sequential(
            nn.Conv1d(d_h, d_hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
        # Output to mels
        self.out = nn.Linear(d_hidden, n_mels)  # [T,192] → [T,80]
    
    def forward(self, x):  # x: [B,T,768] or [T,768]
        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)  # [1,T,768]
        
        B, T, _ = x.shape
        x = self.inp(x)  # [B,T,256]
        x = x.transpose(1, 2)  # [B,256,T] - Conv1d expects [B,C,T]
        x = self.conv(x)  # [B,192,T]
        x = x.transpose(1, 2)  # [B,T,192]
        x = self.out(x)  # [B,T,80]
        
        if not is_batched:
            x = x.squeeze(0)  # [T,80]
        
        return x

class AudioDataset(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Load audio at HiFi-GAN's SR
        wav_hifigan, sr = librosa.load(audio_path, sr=HIFIGAN_SR)
        wav_hifigan = wav_hifigan.astype(np.float32)
        
        # Also load at wav2vec's SR  
        wav_w2v, _ = librosa.load(audio_path, sr=WAV2VEC_SR)
        wav_w2v, _ = librosa.effects.trim(wav_w2v, top_db=20)
        wav_w2v = wav_w2v.astype(np.float32)
        
        # Extract wav2vec layer5 features
        with torch.no_grad():
            inp = self.wav2vec_proc(wav_w2v, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
            out = self.wav2vec_model(**inp)
            wav2vec_feats = out.hidden_states[5].squeeze(0)  # [T_w2v, 768]
        
        # Extract HiFi-GAN mel
        wav_hifigan_tensor = torch.FloatTensor(wav_hifigan).unsqueeze(0)
        mel_hifigan = mel_spectrogram_hifigan(
            wav_hifigan_tensor, N_FFT, N_MELS, HIFIGAN_SR, 
            HOP_SIZE, WIN_SIZE, FMIN, FMAX, center=False
        )  # [1, 80, T_mel]
        
        mel_hifigan = mel_hifigan.squeeze(0).T  # [T_mel, 80]
        
        # Align lengths
        T_mel = mel_hifigan.shape[0]
        T_w2v = wav2vec_feats.shape[0]
        
        if T_w2v != T_mel:
            wav2vec_feats = torch.nn.functional.interpolate(
                wav2vec_feats.T.unsqueeze(0),
                size=T_mel,
                mode='linear',
                align_corners=False
            ).squeeze(0).T
        
        return wav2vec_feats, mel_hifigan

def train_mapper(data_dir, epochs=30, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"🔥 Training Conv1d Mapper (with temporal context!)")
    print(f"HiFi-GAN settings: SR={HIFIGAN_SR}, n_mels={N_MELS}, hop={HOP_SIZE}")
    
    audio_paths = sorted(glob.glob(f"{data_dir}/Actor_*/03-01-*-02-02-01-*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    split = int(0.9 * len(audio_paths))
    train_paths = audio_paths[:split]
    val_paths = audio_paths[split:]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    train_dataset = AudioDataset(train_paths)
    val_dataset = AudioDataset(val_paths)
    
    def collate_fn(batch):
        wav2vec_feats, mel_feats = zip(*batch)
        max_len = max(f.shape[0] for f in wav2vec_feats)
        
        wav2vec_padded = torch.stack([
            torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) 
            for f in wav2vec_feats
        ])
        
        mel_padded = torch.stack([
            torch.nn.functional.pad(m, (0, 0, 0, max_len - m.shape[0]))
            for m in mel_feats
        ])
        
        return wav2vec_padded, mel_padded
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # NEW: Conv1d mapper!
    model = Wav2VecToMelMapperConv().to(device)
    
    # Use L1 loss (better for perceptual quality than MSE)
    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for wav2vec_feats, mel_gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            wav2vec_feats = wav2vec_feats.to(device)
            mel_gt = mel_gt.to(device)
            
            optimizer.zero_grad()
            mel_pred = model(wav2vec_feats)
            loss = criterion(mel_pred, mel_gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for wav2vec_feats, mel_gt in val_loader:
                wav2vec_feats = wav2vec_feats.to(device)
                mel_gt = mel_gt.to(device)
                
                mel_pred = model(wav2vec_feats)
                loss = criterion(mel_pred, mel_gt)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/wav2vec_to_mel_mapper_conv1d.pt')
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
    
    print("\n🎉 Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: models/wav2vec_to_mel_mapper_conv1d.pt")
    print("\nThis should produce MUCH better audio quality!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    
    train_mapper(args.data, args.epochs, args.batch_size, args.lr)

