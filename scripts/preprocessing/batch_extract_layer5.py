#!/usr/bin/env python3
"""
Batch extract wav2vec2 layer5 embeddings for all RAVDESS samples.
Processes happy (03) and angry (05) emotions for specified actors.
"""
import os
import glob
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

def energy_pool(H):
    """Energy-weighted pooling over time."""
    X = H.squeeze(0)                           # [T, 768]
    w = torch.linalg.norm(X, dim=1) + 1e-8     # [T] non-negative weights
    return (X * w[:, None]).sum(dim=0) / w.sum()

def extract_layer5(wav_path, processor, model, device, sr=16000, trim=True):
    """Extract layer5 embedding from a WAV file."""
    wav, _ = librosa.load(wav_path, sr=sr)
    if trim:
        wav, _ = librosa.effects.trim(wav, top_db=20)
    wav = wav.astype(np.float32)
    
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model(**inputs)
    
    H = out.hidden_states[5]  # Layer 5
    emb = energy_pool(H)      # Energy-weighted pooling
    
    return emb.cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch extract layer5 from RAVDESS")
    parser.add_argument("--root", required=True, help="Root directory with Actor_XX folders")
    parser.add_argument("--actors", default="01-24", help="Actor range, e.g., '01-04' or '01-24'")
    parser.add_argument("--emotions", default="03,05", help="Emotion codes: 03=happy, 05=angry")
    parser.add_argument("--out_dir", default="embeddings", help="Output directory")
    args = parser.parse_args()
    
    # Parse actor range
    if "-" in args.actors:
        start, end = args.actors.split("-")
        actor_nums = range(int(start), int(end) + 1)
    else:
        actor_nums = [int(args.actors)]
    
    emotion_codes = args.emotions.split(",")
    emotion_names = {"03": "happy", "05": "angry"}
    
    # Load model once
    print("Loading wav2vec2 model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base",
        output_hidden_states=True
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    
    # Collect all files to process
    files_to_process = []
    
    for actor_num in actor_nums:
        actor_id = f"Actor_{actor_num:02d}"
        actor_dir = os.path.join(args.root, actor_id)
        
        if not os.path.exists(actor_dir):
            continue
        
        wav_files = sorted(glob.glob(os.path.join(actor_dir, "*.wav")))
        
        for wav_path in wav_files:
            basename = os.path.basename(wav_path)
            parts = basename.split("-")
            
            if len(parts) < 7:
                continue
            
            emotion_code = parts[2]
            intensity = parts[3]
            
            if emotion_code not in emotion_codes:
                continue
            
            emotion_name = emotion_names.get(emotion_code, f"emotion{emotion_code}")
            
            # Create output path
            out_subdir = os.path.join(args.out_dir, f"actor{actor_num:02d}")
            os.makedirs(out_subdir, exist_ok=True)
            
            statement = parts[4]
            repetition = parts[5]
            out_name = f"{emotion_name}_i{intensity}_s{statement}_r{repetition}_layer5.npy"
            out_path = os.path.join(out_subdir, out_name)
            
            if os.path.exists(out_path):
                continue
            
            files_to_process.append((wav_path, out_path))
    
    print(f"\nProcessing {len(files_to_process)} files...")
    
    # Process with progress bar
    processed = 0
    failed = []
    
    for wav_path, out_path in tqdm(files_to_process, desc="Extracting layer5"):
        try:
            emb = extract_layer5(wav_path, processor, model, device)
            np.save(out_path, emb)
            processed += 1
        except Exception as e:
            failed.append((wav_path, str(e)))
    
    print("\n" + "="*70)
    print(f"SUMMARY: Processed {processed}/{len(files_to_process)} files")
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for wav, error in failed[:5]:
            print(f"  {wav}: {error[:80]}")
    print("="*70)

if __name__ == "__main__":
    main()

