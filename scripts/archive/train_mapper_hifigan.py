#!/usr/bin/env python3
"""
Train Wav2Vec → Mel Mapper (HiFi-GAN Compatible)
Uses EXACT mel extraction from HiFi-GAN to ensure vocoder compatibility
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

# HiFi-GAN's exact mel extraction (from their meldataset.py)
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

# Same mapper architecture
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

def train_mapper(data_dir, epochs=20, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
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
    
    model = Wav2VecToMelMapper().to(device)
    criterion = nn.MSELoss()
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
            torch.save(model.state_dict(), 'models/wav2vec_to_mel_mapper_hifigan.pt')
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: models/wav2vec_to_mel_mapper_hifigan.pt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    
    train_mapper(args.data, args.epochs, args.batch_size, args.lr)

