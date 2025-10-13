#!/usr/bin/env python3
"""
Train Wav2Vec → Mel Spectrogram Mapper
Simple CNN to map wav2vec layer5 embeddings to mel spectrograms
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

SR = 16000
N_MELS = 80

# Simple CNN Mapper
class Wav2VecToMelMapper(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [batch, seq_len, 768] wav2vec features
        # Output: [batch, seq_len, 80] mel features
        
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
        # x: [batch, seq_len, 768]
        return self.layers(x)  # [batch, seq_len, 80]

class AudioDataset(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Load audio
        wav, sr = librosa.load(audio_path, sr=SR)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        wav = wav.astype(np.float32)
        
        # Extract wav2vec layer5 features
        with torch.no_grad():
            inp = self.wav2vec_proc(wav, sampling_rate=SR, return_tensors="pt", padding=True)
            out = self.wav2vec_model(**inp)
            wav2vec_feats = out.hidden_states[5].squeeze(0)  # [T_w2v, 768]
        
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SR, n_mels=N_MELS, 
            n_fft=1024, hop_length=512, win_length=1024
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # [80, T_mel]
        mel_db = torch.FloatTensor(mel_db.T)  # [T_mel, 80]
        
        # Align lengths (simple: interpolate wav2vec to match mel length)
        T_mel = mel_db.shape[0]
        T_w2v = wav2vec_feats.shape[0]
        
        if T_w2v != T_mel:
            # Interpolate wav2vec features to match mel length
            wav2vec_feats = torch.nn.functional.interpolate(
                wav2vec_feats.T.unsqueeze(0),  # [1, 768, T_w2v]
                size=T_mel,
                mode='linear',
                align_corners=False
            ).squeeze(0).T  # [T_mel, 768]
        
        return wav2vec_feats, mel_db

def train_mapper(data_dir, epochs=20, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Collect audio files
    audio_paths = sorted(glob.glob(f"{data_dir}/Actor_*/03-01-*-02-02-01-*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    # Split train/val
    split = int(0.9 * len(audio_paths))
    train_paths = audio_paths[:split]
    val_paths = audio_paths[split:]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = AudioDataset(train_paths)
    val_dataset = AudioDataset(val_paths)
    
    # Custom collate function to handle variable-length sequences
    def collate_fn(batch):
        wav2vec_feats, mel_feats = zip(*batch)
        
        # Pad to max length in batch
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
    
    # Model, loss, optimizer
    model = Wav2VecToMelMapper().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
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
        
        # Validate
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
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/wav2vec_to_mel_mapper.pt')
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: models/wav2vec_to_mel_mapper.pt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to audio directory (contains Actor_* folders)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    
    train_mapper(args.data, args.epochs, args.batch_size, args.lr)

