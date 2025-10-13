#!/usr/bin/env python3
"""
Validate emotion axis by computing separability metrics.

Usage:
    python validate_axis.py \\
        --axis models/emotion_axis_hubert.npy \\
        --embeddings_root embeddings \\
        --emotion_pair happy angry
"""

import argparse
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_all_embeddings(embeddings_root: str, emotion: str):
    """Load all embeddings for a given emotion across all actors."""
    pattern = f"{embeddings_root}/actor*/{emotion}_hubert.npy"
    files = sorted(glob.glob(pattern))
    
    embeddings = []
    actors = []
    
    for fpath in files:
        emb = np.load(fpath)
        embeddings.append(emb)
        
        # Extract actor ID from path
        actor_id = Path(fpath).parent.name
        actors.append(actor_id)
    
    return np.stack(embeddings, axis=0), actors


def compute_separability(axis, embeddings_a, embeddings_b):
    """
    Compute Cohen's d and classification accuracy.
    
    Args:
        axis: [D,] emotion axis
        embeddings_a: [N, D] embeddings for emotion A
        embeddings_b: [M, D] embeddings for emotion B
        
    Returns:
        dict with metrics
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Project onto axis
    scores_a = embeddings_a @ axis  # [N,]
    scores_b = embeddings_b @ axis  # [M,]
    
    # Cohen's d = (mean_b - mean_a) / pooled_std
    mean_a = scores_a.mean()
    mean_b = scores_b.mean()
    std_a = scores_a.std()
    std_b = scores_b.std()
    
    pooled_std = np.sqrt(0.5 * (std_a**2 + std_b**2))
    cohens_d = (mean_b - mean_a) / (pooled_std + 1e-8)
    
    # Zero-shot accuracy (threshold at midpoint)
    threshold = 0.5 * (mean_a + mean_b)
    acc_a = (scores_a < threshold).mean()  # A should be below threshold
    acc_b = (scores_b > threshold).mean()  # B should be above threshold
    accuracy = 0.5 * (acc_a + acc_b)
    
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
        "cohens_d": cohens_d,
        "threshold": threshold,
        "accuracy": accuracy,
        "scores_a": scores_a,
        "scores_b": scores_b
    }


def compute_per_actor_normalized(axis, embeddings_root, emotion_a, emotion_b):
    """
    Compute per-actor normalized accuracy (z-scoring within each actor).
    
    This removes actor-specific bias (e.g., some speakers naturally project higher).
    """
    pattern_a = f"{embeddings_root}/actor*/{emotion_a}_hubert.npy"
    pattern_b = f"{embeddings_root}/actor*/{emotion_b}_hubert.npy"
    
    files_a = sorted(glob.glob(pattern_a))
    files_b = sorted(glob.glob(pattern_b))
    
    actors_a = {Path(p).parent.name: p for p in files_a}
    actors_b = {Path(p).parent.name: p for p in files_b}
    
    common_actors = sorted(set(actors_a.keys()) & set(actors_b.keys()))
    
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    correct = 0
    total = 0
    
    for actor in common_actors:
        emb_a = np.load(actors_a[actor])
        emb_b = np.load(actors_b[actor])
        
        score_a = float(emb_a @ axis)
        score_b = float(emb_b @ axis)
        
        # Z-score within this actor
        mean_actor = 0.5 * (score_a + score_b)
        std_actor = 0.5 * abs(score_b - score_a)
        
        z_a = (score_a - mean_actor) / (std_actor + 1e-8)
        z_b = (score_b - mean_actor) / (std_actor + 1e-8)
        
        # Classify based on sign of z-score
        # emotion_a should be negative, emotion_b should be positive
        if z_a < 0:
            correct += 1
        if z_b > 0:
            correct += 1
        total += 2
    
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Validate emotion axis")
    parser.add_argument("--axis", required=True, help="Path to emotion axis .npy file")
    parser.add_argument("--embeddings_root", default="embeddings", help="Root embeddings directory")
    parser.add_argument("--emotion_pair", nargs=2, default=["happy", "angry"], 
                       help="Emotion pair (e.g., happy angry)")
    args = parser.parse_args()
    
    emotion_a, emotion_b = args.emotion_pair
    
    print(f"\n{'='*60}")
    print(f"Validating Emotion Axis: {emotion_a} ↔ {emotion_b}")
    print(f"{'='*60}\n")
    
    # Load axis
    axis = np.load(args.axis)
    print(f"[AXIS] Loaded from: {args.axis}")
    print(f"  Shape: {axis.shape}")
    print(f"  Norm: {np.linalg.norm(axis):.6f}")
    
    # Load all embeddings
    embeddings_a, actors_a = load_all_embeddings(args.embeddings_root, emotion_a)
    embeddings_b, actors_b = load_all_embeddings(args.embeddings_root, emotion_b)
    
    print(f"\n[DATA]")
    print(f"  {emotion_a}: {len(embeddings_a)} actors")
    print(f"  {emotion_b}: {len(embeddings_b)} actors")
    
    # Compute separability
    metrics = compute_separability(axis, embeddings_a, embeddings_b)
    
    print(f"\n[PROJECTION SCORES]")
    print(f"  {emotion_a}: {metrics['mean_a']:+.3f} ± {metrics['std_a']:.3f}")
    print(f"  {emotion_b}: {metrics['mean_b']:+.3f} ± {metrics['std_b']:.3f}")
    print(f"  Separation: {abs(metrics['mean_b'] - metrics['mean_a']):.3f}")
    
    print(f"\n[SEPARABILITY]")
    print(f"  Cohen's d: {metrics['cohens_d']:.3f}")
    
    if metrics['cohens_d'] > 1.2:
        print("    ✅ STRONG separation (d > 1.2)")
    elif metrics['cohens_d'] > 0.8:
        print("    ✅ GOOD separation (d > 0.8)")
    elif metrics['cohens_d'] > 0.5:
        print("    ⚠️  MODERATE separation (d > 0.5)")
    else:
        print("    ❌ WEAK separation (d < 0.5)")
    
    print(f"\n[CLASSIFICATION]")
    print(f"  Zero-shot accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Threshold: {metrics['threshold']:+.3f}")
    
    if metrics['accuracy'] > 0.75:
        print("    ✅ STRONG classifier")
    elif metrics['accuracy'] > 0.65:
        print("    ⚠️  MODERATE classifier")
    else:
        print("    ❌ WEAK classifier")
    
    # Per-actor normalized accuracy
    normalized_acc = compute_per_actor_normalized(
        axis, args.embeddings_root, emotion_a, emotion_b
    )
    
    print(f"\n[PER-ACTOR NORMALIZED]")
    print(f"  Accuracy (z-scored): {normalized_acc*100:.1f}%")
    
    if normalized_acc > 0.85:
        print("    ✅ EXCELLENT generalization")
    elif normalized_acc > 0.75:
        print("    ✅ GOOD generalization")
    elif normalized_acc > 0.65:
        print("    ⚠️  MODERATE generalization")
    else:
        print("    ❌ POOR generalization")
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Cohen's d: {metrics['cohens_d']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  Normalized accuracy: {normalized_acc*100:.1f}%")
    
    if metrics['cohens_d'] > 1.0 and normalized_acc > 0.75:
        print("\n  ✅ AXIS IS STRONG - Ready for emotion morphing!")
    elif metrics['cohens_d'] > 0.7 and normalized_acc > 0.65:
        print("\n  ⚠️  AXIS IS MODERATE - May work with careful tuning")
    else:
        print("\n  ❌ AXIS IS WEAK - Consider different feature space")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

