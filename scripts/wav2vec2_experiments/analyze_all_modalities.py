#!/usr/bin/env python3
"""
Compare Layer5, Prosody, and Hybrid across all 24 actors.
Tests within-actor consistency and cross-actor polarity for each modality.
"""
import numpy as np
import glob
import os
from collections import defaultdict

def cosine(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.dot(u, v))

def analyze_modality(happy_files, angry_files, prosody_mean=None, prosody_std=None, modality_name=""):
    """Analyze consistency for one modality (layer5, prosody, or hybrid)."""
    results_by_intensity = {}
    
    for intensity, intensity_name in [('01', 'NORMAL'), ('02', 'STRONG')]:
        actor_means = []
        actor_stds = []
        actor_nums_list = []
        
        for actor_num in sorted(happy_files[intensity].keys()):
            if actor_num not in angry_files[intensity]:
                continue
            
            cosines = []
            
            for h_file in happy_files[intensity][actor_num]:
                for a_file in angry_files[intensity][actor_num]:
                    h = np.load(h_file)
                    a = np.load(a_file)
                    
                    # Normalize prosody if applicable
                    if prosody_mean is not None and prosody_std is not None:
                        h = (h - prosody_mean) / (prosody_std + 1e-6)
                        a = (a - prosody_mean) / (prosody_std + 1e-6)
                    
                    cosines.append(cosine(h, a))
            
            if len(cosines) == 0:
                continue
            
            mean_cos = np.mean(cosines)
            std_cos = np.std(cosines)
            
            actor_means.append(mean_cos)
            actor_stds.append(std_cos)
            actor_nums_list.append(actor_num)
        
        results_by_intensity[intensity] = {
            'means': actor_means,
            'stds': actor_stds,
            'actor_nums': actor_nums_list
        }
    
    # Compute summary stats
    all_within_stds = []
    all_between_stds = []
    all_negative_counts = []
    all_total_counts = []
    
    for intensity in ['01', '02']:
        actor_means = results_by_intensity[intensity]['means']
        actor_stds = results_by_intensity[intensity]['stds']
        
        all_within_stds.extend(actor_stds)
        all_between_stds.append(np.std(actor_means))
        all_negative_counts.append(sum(1 for m in actor_means if m < 0))
        all_total_counts.append(len(actor_means))
    
    avg_within = np.mean(all_within_stds) if all_within_stds else 0
    avg_between = np.mean(all_between_stds) if all_between_stds else 0
    total_negative = sum(all_negative_counts)
    total_actors = sum(all_total_counts)
    
    normal_mean = np.mean(results_by_intensity['01']['means']) if results_by_intensity['01']['means'] else 0
    strong_mean = np.mean(results_by_intensity['02']['means']) if results_by_intensity['02']['means'] else 0
    
    return {
        'within_variance': avg_within,
        'between_variance': avg_between,
        'negative_ratio': total_negative / total_actors if total_actors > 0 else 0,
        'normal_mean': normal_mean,
        'strong_mean': strong_mean,
        'overall_mean': (normal_mean + strong_mean) / 2,
        'results_by_intensity': results_by_intensity
    }

def main():
    # Load all embeddings organized by modality and intensity
    modalities = ['layer5', 'prosody', 'hybrid']
    all_results = {}
    
    # Load prosody normalization
    prosody_mean = np.load('embeddings/stats/prosody_mean.npy')
    prosody_std = np.load('embeddings/stats/prosody_std.npy')
    
    print("="*70)
    print("COMPARING ALL MODALITIES: Layer5 vs Prosody vs Hybrid")
    print("="*70)
    
    for modality in modalities:
        print(f"\n\nProcessing {modality.upper()}...")
        
        happy_files = {'01': defaultdict(list), '02': defaultdict(list)}
        angry_files = {'01': defaultdict(list), '02': defaultdict(list)}
        
        # Collect files
        for actor_num in range(1, 25):
            actor_dir = f"embeddings/actor{actor_num:02d}"
            if not os.path.exists(actor_dir):
                continue
            
            h_files = sorted(glob.glob(os.path.join(actor_dir, f"happy_*_{modality}.npy")))
            a_files = sorted(glob.glob(os.path.join(actor_dir, f"angry_*_{modality}.npy")))
            
            for h_file in h_files:
                basename = os.path.basename(h_file)
                intensity = basename.split('_')[1][1:]  # 'i01' -> '01'
                happy_files[intensity][actor_num].append(h_file)
            
            for a_file in a_files:
                basename = os.path.basename(a_file)
                intensity = basename.split('_')[1][1:]
                angry_files[intensity][actor_num].append(a_file)
        
        # Check if we have data
        total_files = sum(len(happy_files[i][a]) + len(angry_files[i][a]) 
                         for i in ['01', '02'] 
                         for a in happy_files[i].keys())
        
        if total_files == 0:
            print(f"  ⚠️  No {modality} files found - skipping")
            all_results[modality] = None
            continue
        
        print(f"  Found {total_files} files")
        
        # Analyze
        # Only apply prosody normalization to raw prosody features, not hybrid
        use_prosody_norm = (modality == 'prosody')
        results = analyze_modality(
            happy_files, angry_files,
            prosody_mean if use_prosody_norm else None,
            prosody_std if use_prosody_norm else None,
            modality
        )
        
        all_results[modality] = results
    
    # Print comparison
    print("\n\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<30} {'Layer5':>12} {'Prosody':>12} {'Hybrid':>12}")
    print("-" * 70)
    
    metrics = [
        ('Overall mean cosine', 'overall_mean', 'lower'),
        ('Normal intensity mean', 'normal_mean', 'lower'),
        ('Strong intensity mean', 'strong_mean', 'lower'),
        ('Within-actor variance', 'within_variance', 'lower'),
        ('Between-actor variance', 'between_variance', 'higher'),
        ('Negative polarity ratio', 'negative_ratio', 'higher'),
    ]
    
    for metric_name, key, direction in metrics:
        row = f"{metric_name:<30}"
        values = []
        for mod in modalities:
            if all_results[mod] is None:
                row += f"{'N/A':>12}"
                values.append(None)
            else:
                val = all_results[mod][key]
                row += f"{val:>12.3f}"
                values.append(val)
        
        # Mark best value
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        if valid_values:
            if direction == 'lower':
                best_idx = min(valid_values, key=lambda x: x[1])[0]
            else:
                best_idx = max(valid_values, key=lambda x: x[1])[0]
            row += f"  ← {modalities[best_idx].upper()} wins"
        
        print(row)
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Determine winner
    layer5_res = all_results.get('layer5')
    prosody_res = all_results.get('prosody')
    hybrid_res = all_results.get('hybrid')
    
    if all(r is not None for r in [layer5_res, prosody_res, hybrid_res]):
        # Compare overall separation (lower is better)
        layer5_sep = layer5_res['overall_mean']
        prosody_sep = prosody_res['overall_mean']
        hybrid_sep = hybrid_res['overall_mean']
        
        # Compare stability (within-actor variance, lower is better)
        layer5_stable = layer5_res['within_variance']
        prosody_stable = prosody_res['within_variance']
        hybrid_stable = hybrid_res['within_variance']
        
        print(f"\nSeparation (lower = better):")
        print(f"  Layer5:  {layer5_sep:.3f}")
        print(f"  Prosody: {prosody_sep:.3f}")
        print(f"  Hybrid:  {hybrid_sep:.3f}")
        
        print(f"\nStability (lower = better):")
        print(f"  Layer5:  {layer5_stable:.3f}")
        print(f"  Prosody: {prosody_stable:.3f}")
        print(f"  Hybrid:  {hybrid_stable:.3f}")
        
        print("\n" + "-"*70)
        
        if hybrid_sep < layer5_sep and hybrid_stable < 0.2:
            print("✅ HYBRID WINS")
            print("   - Better separation than layer5 alone")
            print("   - Reasonable stability")
            print("   → Use hybrid for emotion detection")
        elif prosody_sep < 0.2 and prosody_stable < 0.15:
            print("✅ PROSODY WINS")
            print("   - Best separation")
            print("   - Good stability")
            print("   → Use prosody-only for emotion detection")
        elif hybrid_stable < prosody_stable and hybrid_sep < prosody_sep:
            print("✅ HYBRID WINS")
            print("   - More stable than prosody")
            print("   - Better separation than prosody")
            print("   → Other LLM was right: hybrid provides alignment")
        else:
            print("⚠️  MIXED RESULTS - No clear winner")
            print("   → Consider fine-tuning wav2vec2 for best results")

if __name__ == "__main__":
    main()

