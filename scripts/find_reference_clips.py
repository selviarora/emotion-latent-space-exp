#!/usr/bin/env python3
"""
Find extreme calm/angry clips using wav2vec2 emotion axis.
These will be reference clips for style-token mixing.
"""
import numpy as np
import torch
import librosa
import glob
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

WAV2VEC_SR = 16000

def extract_wav2vec_features(audio_path):
    """Extract wav2vec2 layer 5 features and compute energy-pooled embedding."""
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
    
    wav, _ = librosa.load(audio_path, sr=WAV2VEC_SR)
    
    inp = proc(wav, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        out = model(**inp)
        H = out.hidden_states[5].squeeze(0)  # [T, 768]
        
        # Energy pooling
        energy = H.pow(2).sum(dim=1) + 1e-8
        weights = energy / energy.sum()
        emb = (H * weights.unsqueeze(1)).sum(dim=0).cpu().numpy()  # [768]
    
    # Normalize
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def main():
    # Load emotion axis (layer5)
    axis = np.load('models/emotion_axis_layer5.npy')
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    print(f"Loaded emotion axis: {axis.shape}")
    
    # Get all audio files
    audio_paths = sorted(glob.glob("/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_*/03-01-*-02-02-01-*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    # Compute axis scores for all files
    print("\nComputing axis scores...")
    scores = []
    for path in audio_paths:
        emb = extract_wav2vec_features(path)
        score = float(np.dot(emb, axis))
        scores.append((path, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1])
    
    print("\n" + "="*80)
    print("TOP 10 MOST CALM (most negative scores - happy/calm end)")
    print("="*80)
    for i, (path, score) in enumerate(scores[:10]):
        filename = path.split('/')[-1]
        actor = path.split('/')[-2]
        print(f"{i+1}. {actor}/{filename}")
        print(f"   Score: {score:+.4f}")
        print(f"   Full path: {path}")
        print()
    
    print("\n" + "="*80)
    print("TOP 10 MOST ANGRY (most positive scores - angry/intense end)")
    print("="*80)
    for i, (path, score) in enumerate(scores[-10:]):
        filename = path.split('/')[-1]
        actor = path.split('/')[-2]
        print(f"{i+1}. {actor}/{filename}")
        print(f"   Score: {score:+.4f}")
        print(f"   Full path: {path}")
        print()
    
    # Save reference list
    ref_data = {
        'calm': scores[:5],  # 5 most calm
        'angry': scores[-5:]  # 5 most angry
    }
    np.save('reference_clips.npy', ref_data)
    print("✅ Saved reference clips to reference_clips.npy")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Listen to a few of these clips to verify they sound calm/angry")
    print("2. Copy 3-5 from each end to a 'references/' folder")
    print("3. Run the style-mixing TTS with these references!")
    print("="*80)

if __name__ == "__main__":
    main()

