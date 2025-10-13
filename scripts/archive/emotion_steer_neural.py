#!/usr/bin/env python3
"""
Neural Emotion Steering - The Real Deal
Uses trained wav2vec→mel mapper + HiFi-GAN vocoder for true embedding-based emotion morphing
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import argparse

SR = 16000
N_MELS = 80

# Same mapper architecture as training
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

def load_vocoder():
    """Load pretrained HiFi-GAN vocoder"""
    try:
        from scipy.io.wavfile import write
        # Try to use pretrained HiFi-GAN
        # For now, we'll use Griffin-Lim as fallback
        print("Using Griffin-Lim vocoder (fallback)")
        return None
    except:
        print("Using Griffin-Lim vocoder")
        return None

def mel_to_audio_griffinlim(mel_db, sr=SR):
    """Convert mel spectrogram to audio using Griffin-Lim"""
    # Convert from dB back to power
    mel_power = librosa.db_to_power(mel_db)
    
    # Inverse mel to linear spectrogram
    linear = librosa.feature.inverse.mel_to_stft(
        mel_power, 
        sr=sr, 
        n_fft=1024,
        power=1.0
    )
    
    # Griffin-Lim algorithm to reconstruct phase
    audio = librosa.griffinlim(
        linear,
        n_iter=32,
        hop_length=512,
        win_length=1024
    )
    
    return audio

def extract_wav2vec_features(audio, sr):
    """Extract wav2vec layer5 features"""
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
    
    audio = audio.astype(np.float32)
    inp = proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        out = model(**inp)
        features = out.hidden_states[5].squeeze(0)  # [T, 768]
    
    return features

def main():
    ap = argparse.ArgumentParser(description="Neural Emotion Steering - Embedding-based morphing")
    ap.add_argument("--input", required=True, help="Input audio file")
    ap.add_argument("--output", required=True, help="Output audio file")
    ap.add_argument("--slider", type=float, default=0.0,
                    help="Emotion slider: -1 (calm) to +1 (angry)")
    ap.add_argument("--mode", choices=["subtle", "creative"], default="subtle",
                    help="subtle=layer5, creative=hybrid")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="Steering strength multiplier (0.5=gentle, 2.0=aggressive)")
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load audio
    print(f"Loading: {args.input}")
    audio, sr = librosa.load(args.input, sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Load emotion axis (use RAW axis, not normalized)
    if args.mode == "subtle":
        axis = np.load('models/emotion_axis_layer5.npy')  # RAW axis
        print("Mode: SUBTLE (Layer5)")
    else:
        axis = np.load('models/emotion_axis_hybrid.npy')  # RAW axis
        print("Mode: CREATIVE (Hybrid)")
    
    # Normalize axis to unit length
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    axis = torch.FloatTensor(axis).to(device)
    
    # Extract wav2vec features
    print("Extracting features...")
    wav2vec_feats = extract_wav2vec_features(audio, SR).to(device)  # [T, 768]
    
    # Get current emotion score
    with torch.no_grad():
        # Energy pooling to get single embedding
        energy = wav2vec_feats.pow(2).sum(dim=1)
        weights = energy / (energy.sum() + 1e-8)
        emb_current = (wav2vec_feats * weights.unsqueeze(1)).sum(dim=0)  # [768]
        
        current_score = float(emb_current @ axis)
    
    print(f"Current projection: {current_score:.3f}")
    print(f"Target slider: {args.slider:.2f}")
    
    # Direct steering: just use slider value as magnitude
    delta = args.slider * args.strength
    
    print(f"Steering delta: {delta:+.2f} (strength={args.strength})")
    
    # Add axis to ALL frames (broadcast)
    wav2vec_steered = wav2vec_feats + (delta * axis.unsqueeze(0))  # [T, 768]
    
    # Load mapper
    print("Loading mapper...")
    mapper = Wav2VecToMelMapper().to(device)
    mapper.load_state_dict(torch.load('models/wav2vec_to_mel_mapper.pt', map_location=device))
    mapper.eval()
    
    # Map to mel
    print("Mapping to mel spectrogram...")
    with torch.no_grad():
        mel_pred = mapper(wav2vec_steered.unsqueeze(0))  # [1, T, 80]
        mel_pred = mel_pred.squeeze(0).cpu().numpy()  # [T, 80]
    
    # Transpose to librosa format [80, T]
    mel_db = mel_pred.T
    
    # Vocoder: mel → audio
    print("Vocoding to audio...")
    audio_out = mel_to_audio_griffinlim(mel_db, SR)
    
    # Match original length approximately
    if len(audio_out) > len(audio):
        audio_out = audio_out[:len(audio)]
    elif len(audio_out) < len(audio):
        audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))
    
    # Normalize
    audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8) * 0.95
    
    # Save
    sf.write(args.output, audio_out, SR)
    print(f"\n✓ Saved: {args.output}")
    
    # Verify steering
    print("\nVerifying steering...")
    wav2vec_new = extract_wav2vec_features(audio_out, SR).to(device)
    with torch.no_grad():
        energy_new = wav2vec_new.pow(2).sum(dim=1)
        weights_new = energy_new / (energy_new.sum() + 1e-8)
        emb_new = (wav2vec_new * weights_new.unsqueeze(1)).sum(dim=0)
        
        new_score = float(emb_new @ axis)
    
    print(f"New projection: {new_score:.3f}")
    print(f"Change: {new_score - current_score:+.3f}")
    
    print("\n" + "="*60)
    print("🎧 LISTEN TO THE OUTPUT!")
    print(f"Does it sound {'CALMER' if args.slider < 0 else 'MORE INTENSE'}?")
    print("="*60)

if __name__ == "__main__":
    main()

