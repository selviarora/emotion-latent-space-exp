#!/usr/bin/env python3
"""
Analyze prosody consistency across all actors and samples.
Tests whether prosody variance is signal (subtypes) or noise (inconsistency).
"""
import numpy as np
import glob
import os
from collections import defaultdict

def cosine(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.dot(u, v))

def main():
    # Load all prosody files, organized by intensity
    # Structure: happy_files[intensity][actor_num] = [file paths]
    happy_files = {'01': defaultdict(list), '02': defaultdict(list)}
    angry_files = {'01': defaultdict(list), '02': defaultdict(list)}
    
    for actor_num in range(1, 25):
        actor_dir = f"embeddings/actor{actor_num:02d}"
        if not os.path.exists(actor_dir):
            continue
        
        h_files = sorted(glob.glob(os.path.join(actor_dir, "happy_*_prosody.npy")))
        a_files = sorted(glob.glob(os.path.join(actor_dir, "angry_*_prosody.npy")))
        
        # Parse intensity from filename: happy_i01_s01_r01_prosody.npy -> intensity='01'
        for h_file in h_files:
            basename = os.path.basename(h_file)
            intensity = basename.split('_')[1][1:]  # 'i01' -> '01'
            happy_files[intensity][actor_num].append(h_file)
        
        for a_file in a_files:
            basename = os.path.basename(a_file)
            intensity = basename.split('_')[1][1:]  # 'i01' -> '01'
            angry_files[intensity][actor_num].append(a_file)
    
    print("="*70)
    print("PROSODY CONSISTENCY ANALYSIS")
    print("="*70)
    
    # Count total actors
    all_actors = set()
    for intensity in ['01', '02']:
        all_actors.update(happy_files[intensity].keys())
    
    print(f"\nLoaded data for {len(all_actors)} actors")
    print(f"  Normal intensity (01): {len(happy_files['01'])} actors")
    print(f"  Strong intensity (02): {len(happy_files['02'])} actors")
    
    sample_actors = sorted(all_actors)[:3]
    for actor_num in sample_actors:
        h01 = len(happy_files['01'].get(actor_num, []))
        a01 = len(angry_files['01'].get(actor_num, []))
        h02 = len(happy_files['02'].get(actor_num, []))
        a02 = len(angry_files['02'].get(actor_num, []))
        print(f"  Actor {actor_num:02d}: normal({h01}h,{a01}a), strong({h02}h,{a02}a)")
    print("  ...")
    
    # Load prosody normalization
    prosody_mean = np.load('embeddings/prosody_mean.npy')
    prosody_std = np.load('embeddings/prosody_std.npy')
    
    print("\n" + "="*70)
    print("TEST 1: Within-Actor Consistency (by Intensity)")
    print("="*70)
    print("Question: Do multiple samples from the same actor give consistent")
    print("          happy-vs-angry cosine similarities?")
    print()
    
    # Analyze by intensity level
    results_by_intensity = {}
    
    for intensity, intensity_name in [('01', 'NORMAL'), ('02', 'STRONG')]:
        print(f"\n--- {intensity_name} Intensity ---")
        
        actor_means = []
        actor_stds = []
        actor_nums_list = []
        
        for actor_num in sorted(happy_files[intensity].keys()):
            if actor_num not in angry_files[intensity]:
                continue
            
            # Compute all pairwise happy-angry cosines for this actor at this intensity
            cosines = []
            
            for h_file in happy_files[intensity][actor_num]:
                for a_file in angry_files[intensity][actor_num]:
                    h = np.load(h_file)
                    a = np.load(a_file)
                    
                    # Normalize
                    h_norm = (h - prosody_mean) / (prosody_std + 1e-6)
                    a_norm = (a - prosody_mean) / (prosody_std + 1e-6)
                    
                    cosines.append(cosine(h_norm, a_norm))
            
            if len(cosines) == 0:
                continue
            
            mean_cos = np.mean(cosines)
            std_cos = np.std(cosines)
            
            actor_means.append(mean_cos)
            actor_stds.append(std_cos)
            actor_nums_list.append(actor_num)
            
            if actor_num <= 4 or actor_num > 22:  # Show first 4 and last 2
                print(f"  Actor {actor_num:02d}: mean={mean_cos:+.3f}, std={std_cos:.3f}  "
                      f"(from {len(cosines)} comparisons)")
            elif actor_num == 5:
                print("  ...")
        
        results_by_intensity[intensity] = {
            'means': actor_means,
            'stds': actor_stds,
            'actor_nums': actor_nums_list
        }
        
        print(f"\nAverage within-actor std: {np.mean(actor_stds):.3f}")
        print("→ Lower is better (means each actor is consistent)")
    
    print("\n" + "="*70)
    print("TEST 2: Cross-Actor Polarity Consistency (by Intensity)")
    print("="*70)
    print("Question: Do all actors show the same polarity (sign)?")
    print()
    
    for intensity, intensity_name in [('01', 'NORMAL'), ('02', 'STRONG')]:
        actor_means = results_by_intensity[intensity]['means']
        
        positive_actors = sum(1 for m in actor_means if m > 0)
        negative_actors = sum(1 for m in actor_means if m < 0)
        
        print(f"\n--- {intensity_name} Intensity ---")
        print(f"  Positive mean (weak separation):  {positive_actors} actors")
        print(f"  Negative mean (strong separation): {negative_actors} actors")
        print(f"  Zero/near-zero:                    {len(actor_means) - positive_actors - negative_actors} actors")
        
        if negative_actors > positive_actors * 2:
            print("  ✅ CONSISTENT POLARITY - Most actors show negative cosine")
        elif positive_actors > negative_actors * 2:
            print("  ⚠️  MIXED POLARITY - Many actors show positive cosine")
        else:
            print("  ❌ INCONSISTENT POLARITY - Actors split between positive/negative")
    
    print("\n" + "="*70)
    print("TEST 3: Magnitude Variance (Subtype Detection)")
    print("="*70)
    print("Question: Is variance in magnitude capturing different expression styles?")
    print()
    
    for intensity, intensity_name in [('01', 'NORMAL'), ('02', 'STRONG')]:
        actor_means = results_by_intensity[intensity]['means']
        actor_nums = results_by_intensity[intensity]['actor_nums']
        
        print(f"\n--- {intensity_name} Intensity ---")
        print(f"  Mean cosine across actors: {np.mean(actor_means):.3f} ± {np.std(actor_means):.3f}")
        print(f"  Range: [{np.min(actor_means):.3f}, {np.max(actor_means):.3f}]")
        
        # Check if variance is primarily in magnitude (not polarity)
        if np.std(actor_means) > 0.2:
            print("  High variance in magnitude detected!")
            print("  → Could indicate different emotion expression styles (subtypes)")
            
            # Find extreme actors
            min_idx = np.argmin(actor_means)
            max_idx = np.argmax(actor_means)
            
            print(f"  Most separated: Actor {actor_nums[min_idx]:02d} (cosine={actor_means[min_idx]:.3f})")
            print(f"  Least separated: Actor {actor_nums[max_idx]:02d} (cosine={actor_means[max_idx]:.3f})")
        else:
            print("  Low variance - actors show similar separation")
    
    print("\n" + "="*70)
    print("SUMMARY & INTERPRETATION")
    print("="*70)
    
    # Aggregate across both intensities
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
    
    avg_within = np.mean(all_within_stds)
    avg_between = np.mean(all_between_stds)
    total_negative = sum(all_negative_counts)
    total_actors = sum(all_total_counts)
    
    print(f"  Within-actor variance (avg):  {avg_within:.3f}  (should be LOW)")
    print(f"  Between-actor variance (avg): {avg_between:.3f}  (can be HIGH)")
    print(f"  Negative polarity ratio:      {total_negative}/{total_actors} = {total_negative/total_actors:.1%}")
    print()
    
    if avg_within < 0.15 and total_negative > total_actors * 0.7:
        print("✅ PROSODY IS RELIABLE SIGNAL")
        print("   - Low within-actor variance (consistent per speaker)")
        print("   - Consistent negative polarity (coherent emotion axis)")
        print("   - Between-actor variance may indicate subtypes")
        print("\n   → CONCLUSION: Prosody-only is good for emotion + subtype detection")
    elif avg_within < 0.15 and total_negative < total_actors * 0.5:
        print("⚠️  PROSODY SHOWS CONSISTENCY BUT WEAK SEPARATION")
        print("   - Low within-actor variance (consistent per speaker)")
        print("   - But many actors show weak/positive separation")
        print("\n   → CONCLUSION: Hybrid might help by adding wav2vec2 alignment")
    else:
        print("❌ PROSODY IS TOO NOISY")
        print("   - High within-actor variance (inconsistent measurements)")
        print("   - Unreliable for generalization")
        print("\n   → CONCLUSION: Use fine-tuned wav2vec2 instead")
    
    # Check if intensity matters
    normal_means = results_by_intensity['01']['means']
    strong_means = results_by_intensity['02']['means']
    
    print("\n" + "="*70)
    print("TEST 4: Does Intensity Matter?")
    print("="*70)
    
    print(f"  Normal intensity mean: {np.mean(normal_means):.3f}")
    print(f"  Strong intensity mean: {np.mean(strong_means):.3f}")
    print(f"  Difference:            {abs(np.mean(normal_means) - np.mean(strong_means)):.3f}")
    print()
    
    if abs(np.mean(normal_means) - np.mean(strong_means)) > 0.15:
        print("⚠️  INTENSITY MATTERS - Normal and strong show different separation")
        print("   → Should train separate models or include intensity as a feature")
    else:
        print("✅ INTENSITY DOESN'T MATTER MUCH - Similar separation in both")
        print("   → Can pool normal and strong samples together")

if __name__ == "__main__":
    main()

