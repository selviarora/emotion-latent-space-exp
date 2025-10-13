#!/usr/bin/env python3
"""
Fine-tune HiFi-GAN vocoder on mapper-predicted mels.
This teaches HiFi-GAN to handle the mapper's specific mel patterns.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import json
from tqdm import tqdm
import soundfile as sf

# HiFi-GAN imports
import sys
hifigan_path = os.path.join(os.path.dirname(__file__), '..', '..', 'hifigan')
sys.path.insert(0, hifigan_path)
from models import Generator
from meldataset import mel_spectrogram

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class MapperMelDataset(Dataset):
    def __init__(self, mel_dir, segment_size=8192):
        self.mel_files = sorted(glob.glob(f"{mel_dir}/*_mel.npy"))
        self.audio_files = [f.replace('_mel.npy', '.wav') for f in self.mel_files]
        self.segment_size = segment_size
        print(f"Found {len(self.mel_files)} mel-audio pairs")
    
    def __len__(self):
        return len(self.mel_files)
    
    def __getitem__(self, idx):
        # Load predicted mel
        mel = np.load(self.mel_files[idx])  # [80, T]
        
        # Load corresponding audio
        audio, sr = sf.read(self.audio_files[idx])
        audio = torch.FloatTensor(audio).unsqueeze(0)  # [1, N]
        
        # Random segment for training (if long enough)
        if audio.shape[1] >= self.segment_size:
            # Corresponding frames
            mel_frames = mel.shape[1]
            audio_frames = audio.shape[1]
            hop_size = 256  # HiFi-GAN hop
            
            # Random start
            max_mel_start = mel_frames - (self.segment_size // hop_size)
            if max_mel_start > 0:
                mel_start = np.random.randint(0, max_mel_start)
                audio_start = mel_start * hop_size
                
                mel = mel[:, mel_start:mel_start + self.segment_size // hop_size]
                audio = audio[:, audio_start:audio_start + self.segment_size]
        
        mel = torch.FloatTensor(mel)
        
        return mel, audio.squeeze(0)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel_dir", default="mapper_mels", help="Directory with predicted mels")
    ap.add_argument("--checkpoint", default="hifigan/generator_universal.pth.tar", help="HiFi-GAN checkpoint to fine-tune")
    ap.add_argument("--config", default="hifigan/config_universal.json", help="HiFi-GAN config")
    ap.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower for fine-tuning)")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load HiFi-GAN config
    with open(args.config) as f:
        config = json.load(f)
    h = AttrDict(config)
    
    # Load generator
    print("Loading HiFi-GAN generator...")
    generator = Generator(h).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(state_dict['generator'])
    
    # Optimizer (lower LR for fine-tuning)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=[0.8, 0.99])
    
    # Dataset & Dataloader
    dataset = MapperMelDataset(args.mel_dir, segment_size=h.segment_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # Loss functions
    l1_loss = nn.L1Loss()
    
    # Fine-tuning loop
    print(f"\n🔥 Fine-tuning HiFi-GAN for {args.epochs} epochs...")
    print(f"   LR: {args.lr} (lower than normal training)")
    
    generator.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for mel, audio in progress:
            mel = mel.to(device)
            audio = audio.to(device).unsqueeze(1)  # [B, 1, T]
            
            # Forward
            audio_pred = generator(mel)
            
            # Match lengths
            min_len = min(audio.shape[2], audio_pred.shape[2])
            audio = audio[:, :, :min_len]
            audio_pred = audio_pred[:, :, :min_len]
            
            # L1 loss (simple but effective for fine-tuning)
            loss = l1_loss(audio_pred, audio)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
        # Save checkpoint every epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = f"models/hifigan_finetuned_epoch{epoch+1}.pt"
            torch.save({
                'generator': generator.state_dict(),
                'epoch': epoch + 1,
            }, ckpt_path)
            print(f"💾 Saved: {ckpt_path}")
    
    # Save final
    final_path = "models/hifigan_finetuned.pt"
    torch.save({
        'generator': generator.state_dict(),
    }, final_path)
    print(f"\n✅ Fine-tuning complete! Saved: {final_path}")
    print(f"   Update emotion_steer_final.py to use this checkpoint")

if __name__ == "__main__":
    main()

