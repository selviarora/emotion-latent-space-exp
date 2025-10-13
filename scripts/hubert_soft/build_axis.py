#!/usr/bin/env python3
"""
Build global emotion axis from per-actor HuBERT embeddings.

Usage:
    python build_axis.py --emotion_pair happy angry --output models/emotion_axis_hubert.npy
"""

import argparse
import glob
import numpy as np
from pathlib import Path


def load_emotion_pairs(embeddings_root: str, emotion_a: str, emotion_b: str):
    """
    Load emotion pairs from all actors.
    
    Args:
        embeddings_root: Path to embeddings/ directory
        emotion_a: First emotion (e.g., "happy")
        emotion_b: Second emotion (e.g., "angry")
        
    Returns:
        axes: [N_actors, 768] per-actor emotion axes (normalized)
        actors: List of actor IDs
    """
    pattern_a = f"{embeddings_root}/actor*/{emotion_a}_hubert.npy"
    pattern_b = f"{embeddings_root}/actor*/{emotion_b}_hubert.npy"
    
    files_a = sorted(glob.glob(pattern_a))
    files_b = sorted(glob.glob(pattern_b))
    
    # Match by actor directory
    actors_a = {Path(p).parent.name: p for p in files_a}
    actors_b = {Path(p).parent.name: p for p in files_b}
    
    common_actors = sorted(set(actors_a.keys()) & set(actors_b.keys()))
    
    if not common_actors:
        print(f"[ERROR] No common actors found for {emotion_a} and {emotion_b}")
        return np.array([]), []
    
    print(f"[INFO] Found {len(common_actors)} actors with both {emotion_a} and {emotion_b}")
    
    axes = []
    for actor in common_actors:
        emb_a = np.load(actors_a[actor])
        emb_b = np.load(actors_b[actor])
        
        # Compute axis: B - A (e.g., angry - happy)
        axis = emb_b - emb_a
        
        # Normalize
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        axes.append(axis)
    
    return np.stack(axes, axis=0), common_actors


def main():
    parser = argparse.ArgumentParser(description="Build emotion axis from HuBERT embeddings")
    parser.add_argument("--embeddings_root", default="embeddings", help="Root embeddings directory")
    parser.add_argument("--emotion_pair", nargs=2, default=["happy", "angry"], 
                       help="Emotion pair (e.g., happy angry)")
    parser.add_argument("--output", default="models/emotion_axis_hubert.npy", help="Output path for axis")
    args = parser.parse_args()
    
    emotion_a, emotion_b = args.emotion_pair
    
    print(f"\n{'='*60}")
    print(f"Building emotion axis: {emotion_b} ← → {emotion_a}")
    print(f"{'='*60}\n")
    
    # Load per-actor axes
    axes, actors = load_emotion_pairs(args.embeddings_root, emotion_a, emotion_b)
    
    if axes.size == 0:
        print("[ERROR] No valid emotion pairs found. Exiting.")
        return
    
    print(f"\n[STATS] Per-actor axes:")
    print(f"  Shape: {axes.shape}")
    print(f"  Actors: {actors}")
    
    # Compute pairwise alignment (cosine similarity between actor axes)
    pairwise_cosine = axes @ axes.T
    off_diagonal = pairwise_cosine[~np.eye(pairwise_cosine.shape[0], dtype=bool)]
    
    print(f"\n[ALIGNMENT] Pairwise cosine similarity:")
    print(f"  Mean: {off_diagonal.mean():.3f}")
    print(f"  Std:  {off_diagonal.std():.3f}")
    print(f"  Min:  {off_diagonal.min():.3f}")
    print(f"  Max:  {off_diagonal.max():.3f}")
    
    # Compute resultant length (measure of consistency)
    mean_unnormalized = axes.mean(axis=0)
    resultant_length = np.linalg.norm(mean_unnormalized)
    
    print(f"\n[RESULTANT] Vector length (0..1): {resultant_length:.3f}")
    if resultant_length > 0.7:
        print("  ✅ Strong consistency across actors")
    elif resultant_length > 0.5:
        print("  ⚠️  Moderate consistency")
    else:
        print("  ❌ Weak consistency - axes are divergent")
    
    # Build global axis (average + normalize)
    global_axis = axes.mean(axis=0)
    global_axis = global_axis / (np.linalg.norm(global_axis) + 1e-8)
    
    print(f"\n[GLOBAL AXIS]")
    print(f"  Shape: {global_axis.shape}")
    print(f"  Norm: {np.linalg.norm(global_axis):.6f} (should be ~1.0)")
    print(f"  Direction: {emotion_a} → {emotion_b}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, global_axis)
    
    print(f"\n[SAVE] Emotion axis saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

