#!/usr/bin/env python3
"""
Train Wav2Vec → Mel Mapper - FINAL VERSION
With residuals, dilations, LayerNorm, and delta losses
Based on expert recommendations
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
from librosa.filters import mel as librosa_mel_fn

# HiFi-GAN exact settings
HIFIGAN_SR = 22050
N_FFT = 1024
N_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000
WAV2VEC_SR = 16000

# HiFi-GAN's exact mel extraction
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

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
    spec = dynamic_range_compression_torch(spec)

    return spec

# Expert's architecture with residuals + dilations + LayerNorm
class ConvBlock(nn.Module):
    def __init__(self, ch, k=7, d=1):
        super().__init__()
        pad = (k-1)//2 * d
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, dilation=d, padding=pad)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(ch)
    
    def forward(self, x):  # x: [B, C, T]
        y = self.conv(x)
        y = self.act(y)
        # LayerNorm expects [B, T, C]
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + y  # Residual connection

class Wav2VecToMelTemporal(nn.Module):
    def __init__(self, in_ch=768, hidden=512, n_mels=80):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        # Residual blocks with increasing dilation
        self.blocks = nn.Sequential(
            ConvBlock(hidden, k=7, d=1),
            ConvBlock(hidden, k=7, d=2),
            ConvBlock(hidden, k=7, d=4),
            ConvBlock(hidden, k=7, d=8),   # more temporal context
            ConvBlock(hidden, k=7, d=16),  # even more
        )
        self.proj = nn.Conv1d(hidden, n_mels, 1)

    def forward(self, X):  # X: [B, T, 768]
        X = X.transpose(1, 2)  # [B, 768, T]
        H = self.inp(X)
        H = self.blocks(H)
        M = self.proj(H)  # [B, 80, T]
        return M.transpose(1, 2)  # [B, T, 80]

class AudioDataset(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Load audio ONCE at original SR
        audio_orig, sr_orig = librosa.load(audio_path, sr=None)
        
        # NO TRIMMING - to ensure perfect temporal alignment
        # (Trimming can cause subtle misalignment between wav2vec and mel frames)
        
        # Resample to both target SRs
        wav_hifigan = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=HIFIGAN_SR)
        wav_w2v = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=WAV2VEC_SR)
        
        wav_hifigan = wav_hifigan.astype(np.float32)
        wav_w2v = wav_w2v.astype(np.float32)
        
        # RMS normalization for consistency
        def rms_norm(x, target=0.1, eps=1e-8):
            r = np.sqrt((x**2).mean() + eps)
            return x * (target / (r + eps))
        
        wav_hifigan = rms_norm(wav_hifigan)
        wav_w2v = rms_norm(wav_w2v)
        
        # Extract wav2vec layer5 features
        with torch.no_grad():
            inp = self.wav2vec_proc(wav_w2v, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
            out = self.wav2vec_model(**inp)
            wav2vec_feats = out.hidden_states[5].squeeze(0)
        
        # Extract HiFi-GAN mel FIRST to get EXACT frame count
        wav_hifigan_tensor = torch.FloatTensor(wav_hifigan).unsqueeze(0)
        mel_hifigan = mel_spectrogram_hifigan(
            wav_hifigan_tensor, N_FFT, N_MELS, HIFIGAN_SR, 
            HOP_SIZE, WIN_SIZE, FMIN, FMAX, center=False
        )
        
        mel_hifigan = mel_hifigan.squeeze(0).T  # [T_mel_actual, 80]
        
        # Use ACTUAL mel frame count (not calculated)
        T_mel_actual = mel_hifigan.shape[0]
        T_w2v = wav2vec_feats.shape[0]
        
        # Always upsample to match EXACT GT mel length
        if T_w2v != T_mel_actual:
            wav2vec_feats = torch.nn.functional.interpolate(
                wav2vec_feats.T.unsqueeze(0),
                size=T_mel_actual,
                mode='linear',
                align_corners=False
            ).squeeze(0).T
        
        return wav2vec_feats, mel_hifigan

def train_mapper(data_dir, epochs=30, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"🔥 Training FINAL Mapper (Residual + Dilation + Delta Losses)")
    print(f"HiFi-GAN settings: SR={HIFIGAN_SR}, n_mels={N_MELS}, hop={HOP_SIZE}")
    
    # Loss logging
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"training_log_{timestamp}.csv"
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
    print(f"📊 Logging to: {log_file}")
    
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
        lengths = torch.LongTensor([f.shape[0] for f in wav2vec_feats])
        max_len = lengths.max().item()
        
        wav2vec_padded = torch.stack([
            torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) 
            for f in wav2vec_feats
        ])
        
        mel_padded = torch.stack([
            torch.nn.functional.pad(m, (0, 0, 0, max_len - m.shape[0]))
            for m in mel_feats
        ])
        
        return wav2vec_padded, mel_padded, lengths
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Expert's architecture
    model = Wav2VecToMelTemporal(in_ch=768, hidden=512, n_mels=N_MELS).to(device)
    
    # Combined loss with padding mask: L1 + delta + delta-delta
    def combined_loss(pred, target, lengths, w0=1.0, w1=0.8, w2=0.4):  # stronger temporal detail
        B, T, C = pred.shape
        # Create mask: [B, T, 1]
        mask = torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1).to(pred.device)
        mask = mask.unsqueeze(2).float()  # [B, T, 1]
        
        # Masked L1 loss
        l0 = (torch.abs(pred - target) * mask).sum() / mask.sum()
        
        # Masked delta losses (temporal derivatives)
        pred_d1 = pred[:, 1:, :] - pred[:, :-1, :]
        target_d1 = target[:, 1:, :] - target[:, :-1, :]
        mask_d1 = mask[:, 1:, :]
        l1 = (torch.abs(pred_d1 - target_d1) * mask_d1).sum() / mask_d1.sum()
        
        pred_d2 = pred_d1[:, 1:, :] - pred_d1[:, :-1, :]
        target_d2 = target_d1[:, 1:, :] - target_d1[:, :-1, :]
        mask_d2 = mask[:, 2:, :]
        l2 = (torch.abs(pred_d2 - target_d2) * mask_d2).sum() / mask_d2.sum()
        
        return w0 * l0 + w1 * l1 + w2 * l2
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Print shapes once per epoch
        if epoch == 0:
            for wav2vec_feats, mel_gt, lengths in train_loader:
                print(f"\n📊 Batch shapes (epoch {epoch+1}):")
                print(f"  w2v: {wav2vec_feats.shape}  mel: {mel_gt.shape}")
                print(f"  T matches: {wav2vec_feats.shape[1] == mel_gt.shape[1]} ✓" if wav2vec_feats.shape[1] == mel_gt.shape[1] else f"  ⚠️ T MISMATCH!")
                break
        
        for batch_idx, (wav2vec_feats, mel_gt, lengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            wav2vec_feats = wav2vec_feats.to(device)
            mel_gt = mel_gt.to(device)
            
            optimizer.zero_grad()
            mel_pred = model(wav2vec_feats)
            
            # Print mel stats on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                mp = mel_pred[0]
                mg = mel_gt[0]
                print(f"\n📈 Mel stats (first batch, epoch 1):")
                print(f"  pred mean/std/min/max: {mp.mean().item():.3f} / {mp.std().item():.3f} / {mp.min().item():.3f} / {mp.max().item():.3f}")
                print(f"  gt   mean/std/min/max: {mg.mean().item():.3f} / {mg.std().item():.3f} / {mg.min().item():.3f} / {mg.max().item():.3f}")
            
            loss = combined_loss(mel_pred, mel_gt, lengths)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for wav2vec_feats, mel_gt, lengths in val_loader:
                wav2vec_feats = wav2vec_feats.to(device)
                mel_gt = mel_gt.to(device)
                
                mel_pred = model(wav2vec_feats)
                loss = combined_loss(mel_pred, mel_gt, lengths)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/wav2vec_to_mel_mapper_final.pt')
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
    
    print("\n🎉 Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: models/wav2vec_to_mel_mapper_final.pt")
    print("\n✨ WITH PADDING MASKING - This should ACTUALLY work now!")
    print("The model is no longer learning to predict zeros for padding!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    
    train_mapper(args.data, args.epochs, args.batch_size, args.lr)

