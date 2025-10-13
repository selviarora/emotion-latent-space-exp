#!/usr/bin/env python3
"""
Batch create hybrid embeddings by combining layer5 + prosody.
Requires that both layer5 and prosody files already exist.
"""
import os
import glob
import numpy as np
from tqdm import tqdm

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch create hybrid embeddings")
    parser.add_argument("--actors", default="01-24", help="Actor range, e.g., '01-04' or '01-24'")
    parser.add_argument("--embeddings_dir", default="embeddings", help="Embeddings directory")
    parser.add_argument("--prosody_mean", help="Path to prosody mean (optional)")
    parser.add_argument("--prosody_std", help="Path to prosody std (optional)")
    args = parser.parse_args()
    
    # Parse actor range
    if "-" in args.actors:
        start, end = args.actors.split("-")
        actor_nums = range(int(start), int(end) + 1)
    else:
        actor_nums = [int(args.actors)]
    
    # Load normalization if provided
    prosody_mean = None
    prosody_std = None
    if args.prosody_mean and args.prosody_std:
        prosody_mean = np.load(args.prosody_mean)
        prosody_std = np.load(args.prosody_std)
        print("Using prosody normalization (z-score)")
    else:
        # Load default if exists
        default_mean = os.path.join(args.embeddings_dir, "stats", "prosody_mean.npy")
        default_std = os.path.join(args.embeddings_dir, "stats", "prosody_std.npy")
        if os.path.exists(default_mean) and os.path.exists(default_std):
            prosody_mean = np.load(default_mean)
            prosody_std = np.load(default_std)
            print("Using default prosody normalization (z-score)")
        else:
            print("No normalization - using adaptive scaling")
    
    # Collect all layer5/prosody pairs
    pairs_to_process = []
    
    for actor_num in actor_nums:
        actor_dir = os.path.join(args.embeddings_dir, f"actor{actor_num:02d}")
        
        if not os.path.exists(actor_dir):
            continue
        
        # Find all layer5 files
        layer5_files = sorted(glob.glob(os.path.join(actor_dir, "*_layer5.npy")))
        
        for l5_path in layer5_files:
            # Derive prosody path
            basename = os.path.basename(l5_path)
            prosody_name = basename.replace("_layer5.npy", "_prosody.npy")
            prosody_path = os.path.join(actor_dir, prosody_name)
            
            if not os.path.exists(prosody_path):
                continue
            
            # Derive hybrid output path
            hybrid_name = basename.replace("_layer5.npy", "_hybrid.npy")
            hybrid_path = os.path.join(actor_dir, hybrid_name)
            
            if os.path.exists(hybrid_path):
                continue
            
            pairs_to_process.append((l5_path, prosody_path, hybrid_path))
    
    print(f"\nProcessing {len(pairs_to_process)} hybrid embeddings...")
    
    processed = 0
    failed = []
    
    for l5_path, prosody_path, hybrid_path in tqdm(pairs_to_process, desc="Creating hybrids"):
        try:
            e = np.load(l5_path)       # (768,)
            p = np.load(prosody_path)  # (8,)
            
            # Normalize prosody
            if prosody_mean is not None and prosody_std is not None:
                # Z-score normalization
                p = (p - prosody_mean) / (prosody_std + 1e-6)
            else:
                # Adaptive scaling to match wav2vec2 magnitude
                p = p / (np.linalg.norm(p) + 1e-9)
                p = p * np.linalg.norm(e) / np.sqrt(768) * np.sqrt(8)
            
            h = np.concatenate([e, p], axis=0)  # (776,)
            np.save(hybrid_path, h)
            processed += 1
            
        except Exception as ex:
            failed.append((l5_path, str(ex)))
    
    print("\n" + "="*70)
    print(f"SUMMARY: Created {processed}/{len(pairs_to_process)} hybrid embeddings")
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for path, error in failed[:5]:
            print(f"  {path}: {error[:80]}")
    print("="*70)

if __name__ == "__main__":
    main()

