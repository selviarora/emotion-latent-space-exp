import numpy as np
import argparse
from sklearn.metrics import silhouette_score
import torch, librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import glob

SR = 16000
proc = None
model = None

def embed_layer5_energy(wav_path, layer_index=5):
    global proc, model
    if proc is None:
        proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    if model is None:
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

    wav, sr = librosa.load(wav_path, sr=SR)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    wav = wav.astype(np.float32)

    inp = proc(wav, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inp)
        H = out.hidden_states[layer_index].squeeze(0)
        w = (H.pow(2).sum(dim=1) + 1e-8); w = w / w.sum()
        emb = (H * w.unsqueeze(1)).sum(dim=0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb)+1e-8)
    return emb

def create_hybrid_embedding(wav_path):
    """Create hybrid embedding (layer5 + prosody)"""
    # Layer5
    emb = embed_layer5_energy(wav_path, layer_index=5)
    
    # Prosody
    prosody_mean = np.load('embeddings/stats/prosody_mean.npy')
    prosody_std = np.load('embeddings/stats/prosody_std.npy')
    
    wav, sr = librosa.load(wav_path, sr=SR)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    
    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
    f0 = librosa.yin(wav, fmin=80, fmax=400, sr=sr)
    f0_valid = f0[f0 > 0]
    
    prosody_feats = np.array([
        np.mean(f0_valid) if len(f0_valid) > 0 else 0,
        np.std(f0_valid) if len(f0_valid) > 0 else 0,
        np.ptp(f0_valid) if len(f0_valid) > 0 else 0,
        np.mean(rms),
        np.std(rms),
        np.ptp(rms),
        len(wav) / sr,
        sr / len(wav) if len(wav) > 0 else 0
    ])
    
    prosody_feats = (prosody_feats - prosody_mean) / (prosody_std + 1e-6)
    
    return np.concatenate([emb, prosody_feats])

def analyze_axis(axis, embeddings_calm, embeddings_angry, label=""):
    """Compute separability metrics for an axis"""
    
    # Project onto axis
    axis_norm = axis / (np.linalg.norm(axis) + 1e-8)
    
    calm_scores = [float(np.dot(e, axis_norm)) for e in embeddings_calm]
    angry_scores = [float(np.dot(e, axis_norm)) for e in embeddings_angry]
    
    # Combine for silhouette
    all_scores = np.array(calm_scores + angry_scores).reshape(-1, 1)
    labels = np.array([0]*len(calm_scores) + [1]*len(angry_scores))
    
    # Silhouette score
    if len(set(labels)) > 1:
        silhouette = silhouette_score(all_scores, labels)
    else:
        silhouette = 0.0
    
    # Cohen's d (effect size)
    calm_arr = np.array(calm_scores)
    angry_arr = np.array(angry_scores)
    
    pooled_std = np.sqrt(0.5 * (calm_arr.var() + angry_arr.var()))
    cohens_d = (angry_arr.mean() - calm_arr.mean()) / (pooled_std + 1e-8)
    
    # Means for normalization
    calm_mean = calm_arr.mean()
    angry_mean = angry_arr.mean()
    
    return {
        'silhouette': silhouette,
        'cohens_d': cohens_d,
        'calm_mean': calm_mean,
        'angry_mean': angry_mean,
        'calm_std': calm_arr.std(),
        'angry_std': angry_arr.std(),
        'calm_scores': calm_scores,
        'angry_scores': angry_scores
    }

def normalize_axis(axis, calm_mean, angry_mean):
    """Normalize axis so calm=-1, angry=+1"""
    
    axis_norm = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Current range: [calm_mean, angry_mean]
    # Want: [-1, +1]
    
    # Scale factor: distance from calm to angry should map to 2 (from -1 to +1)
    scale = 2.0 / (angry_mean - calm_mean + 1e-8)
    
    # Shift: calm_mean should map to -1
    # After scaling: calm_mean * scale = -1
    # So we need: axis_normalized such that dot(calm_centroid, axis_normalized) = -1
    
    # Simple approach: scale the axis
    axis_normalized = axis_norm * scale
    
    # Verify the midpoint
    midpoint = (calm_mean + angry_mean) / 2
    # After normalization, midpoint should be 0
    # New midpoint = (midpoint - calm_mean) * scale - 1 = 0
    # This means: (midpoint - calm_mean) * scale = 1
    
    # Actually, simpler: shift then scale
    # projection_new = (projection_old - midpoint) * scale
    # But we want to bake this into the axis itself...
    
    # Even simpler: use the fact that we can add a bias term later
    # Just scale the axis so the RANGE is 2
    axis_normalized = axis_norm * scale
    
    return axis_normalized, scale, (calm_mean + angry_mean) / 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Audio root directory")
    args = ap.parse_args()
    
    # Load axes
    axis_layer5 = np.load('emotion_axis_layer5.npy')
    axis_hybrid = np.load('emotion_axis_hybrid.npy')
    
    # Collect embeddings
    calm_paths = sorted(glob.glob(f"{args.root}/Actor_*/03-01-02-02-02-01-*.wav"))  # 02 = calm
    angry_paths = sorted(glob.glob(f"{args.root}/Actor_*/03-01-05-02-02-01-*.wav"))  # 05 = angry
    
    print(f"Found {len(calm_paths)} calm files, {len(angry_paths)} angry files")
    print("\nExtracting embeddings...")
    
    # Layer5 embeddings
    print("  Layer5...")
    embeddings_calm_l5 = [embed_layer5_energy(p) for p in calm_paths]
    embeddings_angry_l5 = [embed_layer5_energy(p) for p in angry_paths]
    
    # Hybrid embeddings
    print("  Hybrid...")
    embeddings_calm_hyb = [create_hybrid_embedding(p) for p in calm_paths]
    embeddings_angry_hyb = [create_hybrid_embedding(p) for p in angry_paths]
    
    # Analyze both
    print("\n" + "="*70)
    print("AXIS COMPARISON: Layer5 vs Hybrid")
    print("="*70)
    
    results_l5 = analyze_axis(axis_layer5, embeddings_calm_l5, embeddings_angry_l5, "Layer5")
    results_hyb = analyze_axis(axis_hybrid, embeddings_calm_hyb, embeddings_angry_hyb, "Hybrid")
    
    print(f"\n{'Metric':<30} {'Layer5':>15} {'Hybrid':>15} {'Winner':>15}")
    print("-"*70)
    
    metrics = [
        ('Silhouette Score', 'silhouette', 'higher', '.3f'),
        ("Cohen's d (effect size)", 'cohens_d', 'higher', '.3f'),
        ('Calm mean projection', 'calm_mean', 'lower', '.3f'),
        ('Angry mean projection', 'angry_mean', 'higher', '.3f'),
        ('Calm std (consistency)', 'calm_std', 'lower', '.3f'),
        ('Angry std (consistency)', 'angry_std', 'lower', '.3f'),
    ]
    
    for metric_name, key, direction, fmt in metrics:
        val_l5 = results_l5[key]
        val_hyb = results_hyb[key]
        
        if direction == 'higher':
            winner = "Layer5" if val_l5 > val_hyb else "Hybrid"
        else:
            winner = "Layer5" if val_l5 < val_hyb else "Hybrid"
        
        print(f"{metric_name:<30} {val_l5:>15{fmt}} {val_hyb:>15{fmt}} {winner:>15}")
    
    # Normalize both axes
    print("\n" + "="*70)
    print("NORMALIZING AXES (calm=-1, angry=+1)")
    print("="*70)
    
    axis_l5_norm, scale_l5, mid_l5 = normalize_axis(
        axis_layer5, 
        results_l5['calm_mean'], 
        results_l5['angry_mean']
    )
    
    axis_hyb_norm, scale_hyb, mid_hyb = normalize_axis(
        axis_hybrid,
        results_hyb['calm_mean'],
        results_hyb['angry_mean']
    )
    
    print(f"\nLayer5:")
    print(f"  Scale factor: {scale_l5:.3f}")
    print(f"  Midpoint shift: {mid_l5:.3f}")
    print(f"  Calm will project to: ~{(results_l5['calm_mean'] - mid_l5) * scale_l5:.3f}")
    print(f"  Angry will project to: ~{(results_l5['angry_mean'] - mid_l5) * scale_l5:.3f}")
    
    print(f"\nHybrid:")
    print(f"  Scale factor: {scale_hyb:.3f}")
    print(f"  Midpoint shift: {mid_hyb:.3f}")
    print(f"  Calm will project to: ~{(results_hyb['calm_mean'] - mid_hyb) * scale_hyb:.3f}")
    print(f"  Angry will project to: ~{(results_hyb['angry_mean'] - mid_hyb) * scale_hyb:.3f}")
    
    # Save normalized axes
    np.save('emotion_axis_layer5_normalized.npy', axis_l5_norm)
    np.save('emotion_axis_hybrid_normalized.npy', axis_hyb_norm)
    
    # Save metadata
    np.save('emotion_axis_layer5_metadata.npy', np.array([scale_l5, mid_l5]))
    np.save('emotion_axis_hybrid_metadata.npy', np.array([scale_hyb, mid_hyb]))
    
    print("\n" + "="*70)
    print("SAVED:")
    print("  emotion_axis_layer5_normalized.npy")
    print("  emotion_axis_hybrid_normalized.npy")
    print("  emotion_axis_layer5_metadata.npy (scale, midpoint)")
    print("  emotion_axis_hybrid_metadata.npy (scale, midpoint)")
    print("="*70)
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    
    # Decide winner
    if results_hyb['silhouette'] > results_l5['silhouette'] and results_hyb['cohens_d'] > results_l5['cohens_d']:
        print("✅ USE HYBRID for slider")
        print(f"   - Better silhouette: {results_hyb['silhouette']:.3f} vs {results_l5['silhouette']:.3f}")
        print(f"   - Better effect size: {results_hyb['cohens_d']:.3f} vs {results_l5['cohens_d']:.3f}")
        print(f"   - Cleaner gradient for slider control")
    elif results_l5['silhouette'] > results_hyb['silhouette'] and results_l5['cohens_d'] > results_hyb['cohens_d']:
        print("✅ USE LAYER5 for slider")
        print(f"   - Better silhouette: {results_l5['silhouette']:.3f} vs {results_hyb['silhouette']:.3f}")
        print(f"   - Better effect size: {results_l5['cohens_d']:.3f} vs {results_hyb['cohens_d']:.3f}")
        print(f"   - Simpler (no prosody extraction needed)")
    else:
        print("⚖️  MIXED RESULTS - Try both!")
        print(f"   Layer5 silhouette: {results_l5['silhouette']:.3f}, d: {results_l5['cohens_d']:.3f}")
        print(f"   Hybrid silhouette: {results_hyb['silhouette']:.3f}, d: {results_hyb['cohens_d']:.3f}")
    
    print("\nUSAGE for slider:")
    print("  score = dot(embedding, normalized_axis) - midpoint")
    print("  score will range roughly [-1, +1]")
    print("  -1 = calm, 0 = neutral, +1 = angry")
    print("="*70)

if __name__ == "__main__":
    main()

