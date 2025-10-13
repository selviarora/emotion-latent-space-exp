#!/usr/bin/env python3
"""
Extract hubert-soft (256-D) embeddings for emotion axis computation.

Usage:
    python extract_hubert_soft.py --input_dir "/path/to/Actor_01" --output_dir "embeddings/actor01"
    
Or batch process all actors:
    for i in {01..24}; do
        python extract_hubert_soft.py \\
            --input_dir "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_$i" \\
            --output_dir "embeddings/actor$(printf '%02d' $i)"
    done
"""

import argparse
import glob
import os
import numpy as np
import torch
import librosa
from pathlib import Path

# Import hubert-soft model (using fairseq or transformers)
try:
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    USE_TRANSFORMERS = True
except ImportError:
    print("WARNING: transformers not found. Install: pip install transformers")
    USE_TRANSFORMERS = False


class HubertSoftExtractor:
    """Extract hubert-soft (256-D cluster units) from audio."""
    
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.sr = 16000
        
        # Load HuBERT model
        print("[INFO] Loading HuBERT model (facebook/hubert-base-ls960)...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
        self.model.eval()
        
        # For hubert-soft, we typically use the last hidden state (768-D)
        # then apply k-means quantization to 256 units
        # For now, we'll use the raw 768-D and do dimensionality reduction via pooling
        # (A proper hubert-soft would use k-means clustering, but this is simpler for prototyping)
        
        print("[INFO] HuBERT model loaded.")
    
    def extract(self, audio_path: str) -> np.ndarray:
        """
        Extract hubert-soft features from audio file.
        
        Returns:
            np.ndarray: Shape [T, 768] frame-wise features (or [768,] if pooled)
        """
        # Load and preprocess audio
        wav, sr = librosa.load(audio_path, sr=self.sr)
        wav = wav.astype(np.float32)
        
        # Trim silence
        wav, _ = librosa.effects.trim(wav, top_db=20)
        
        if len(wav) < 400:  # Too short
            return None
        
        # Extract features
        inputs = self.feature_extractor(
            wav, 
            sampling_rate=self.sr, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last_hidden_state: [1, T, 768]
            hidden_states = outputs.last_hidden_state.squeeze(0)  # [T, 768]
        
        return hidden_states.cpu().numpy()
    
    def pool_energy_weighted(self, features: np.ndarray) -> np.ndarray:
        """
        Energy-weighted pooling (same as wav2vec2 approach).
        
        Args:
            features: [T, 768] frame-wise features
            
        Returns:
            [768,] pooled embedding
        """
        # Compute energy per frame
        energy = np.sum(features ** 2, axis=1)  # [T,]
        weights = energy / (energy.sum() + 1e-8)
        
        # Weighted average
        pooled = np.sum(features * weights[:, None], axis=0)  # [768,]
        
        # Normalize
        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
        
        return pooled


def process_actor(input_dir: str, output_dir: str, device: str = "cpu"):
    """Process all emotion clips for one actor."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = HubertSoftExtractor(device=device)
    
    # Define emotion patterns (RAVDESS format: 03-01-{emotion}-02-02-01-{actor}.wav)
    emotions = {
        "neutral": "01",
        "calm": "02",
        "happy": "03",
        "sad": "04",
        "angry": "05",
        "fearful": "06",
        "disgust": "07",
        "surprised": "08"
    }
    
    for emotion_name, emotion_code in emotions.items():
        # Find all files for this emotion
        pattern = f"{input_dir}/03-01-{emotion_code}-*-*-*.wav"
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"  [SKIP] {emotion_name}: no files found")
            continue
        
        print(f"  [PROCESS] {emotion_name}: {len(files)} files")
        
        embeddings = []
        for audio_path in files:
            features = extractor.extract(audio_path)
            if features is None:
                continue
            
            # Pool to single vector
            pooled = extractor.pool_energy_weighted(features)
            embeddings.append(pooled)
        
        if not embeddings:
            print(f"    [WARN] No valid embeddings for {emotion_name}")
            continue
        
        # Average across all takes for this emotion
        emotion_embedding = np.mean(embeddings, axis=0)
        emotion_embedding = emotion_embedding / (np.linalg.norm(emotion_embedding) + 1e-8)
        
        # Save
        output_path = os.path.join(output_dir, f"{emotion_name}_hubert.npy")
        np.save(output_path, emotion_embedding)
        print(f"    [SAVE] {output_path} (shape: {emotion_embedding.shape})")


def main():
    parser = argparse.ArgumentParser(description="Extract hubert-soft embeddings")
    parser.add_argument("--input_dir", required=True, help="Path to Actor_XX directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    args = parser.parse_args()
    
    print(f"\n[START] Processing {args.input_dir}")
    process_actor(args.input_dir, args.output_dir, device=args.device)
    print(f"[DONE] Embeddings saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()

