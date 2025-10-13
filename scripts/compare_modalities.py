import numpy as np

def cosine(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.dot(u, v))

actors = ['actor01', 'actor02', 'actor03', 'actor04']

# Load prosody normalization
prosody_mean = np.load('embeddings/prosody_mean.npy')
prosody_std = np.load('embeddings/prosody_std.npy')

print("="*70)
print("COMPARING: Layer5 only vs Prosody only vs Hybrid")
print("="*70)

for actor in actors:
    print(f"\n{actor.upper()}:")
    print("-" * 50)
    
    # Load embeddings
    h_l5 = np.load(f'embeddings/{actor}/happy_layer5.npy')
    a_l5 = np.load(f'embeddings/{actor}/angry_layer5.npy')
    
    h_pros = np.load(f'embeddings/{actor}/happy_prosody.npy')
    a_pros = np.load(f'embeddings/{actor}/angry_prosody.npy')
    
    h_hyb = np.load(f'embeddings/{actor}/happy_hybrid.npy')
    a_hyb = np.load(f'embeddings/{actor}/angry_hybrid.npy')
    
    # Normalize prosody
    h_pros_norm = (h_pros - prosody_mean) / (prosody_std + 1e-6)
    a_pros_norm = (a_pros - prosody_mean) / (prosody_std + 1e-6)
    
    # Compute similarities
    cos_l5 = cosine(h_l5, a_l5)
    cos_pros = cosine(h_pros_norm, a_pros_norm)
    cos_hyb = cosine(h_hyb, a_hyb)
    
    print(f"  Layer5 only:      {cos_l5:.3f}")
    print(f"  Prosody only:     {cos_pros:.3f}")
    print(f"  Hybrid:           {cos_hyb:.3f}")
    print(f"  → Improvement:    {cos_l5 - cos_hyb:+.3f}")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

# Compute averages
l5_sims = []
pros_sims = []
hyb_sims = []

for actor in actors:
    h_l5 = np.load(f'embeddings/{actor}/happy_layer5.npy')
    a_l5 = np.load(f'embeddings/{actor}/angry_layer5.npy')
    h_pros = np.load(f'embeddings/{actor}/happy_prosody.npy')
    a_pros = np.load(f'embeddings/{actor}/angry_prosody.npy')
    h_hyb = np.load(f'embeddings/{actor}/happy_hybrid.npy')
    a_hyb = np.load(f'embeddings/{actor}/angry_hybrid.npy')
    
    h_pros_norm = (h_pros - prosody_mean) / (prosody_std + 1e-6)
    a_pros_norm = (a_pros - prosody_mean) / (prosody_std + 1e-6)
    
    l5_sims.append(cosine(h_l5, a_l5))
    pros_sims.append(cosine(h_pros_norm, a_pros_norm))
    hyb_sims.append(cosine(h_hyb, a_hyb))

print(f"Average same-speaker cross-emotion cosine:")
print(f"  Layer5 only:   {np.mean(l5_sims):.3f} ± {np.std(l5_sims):.3f}")
print(f"  Prosody only:  {np.mean(pros_sims):.3f} ± {np.std(pros_sims):.3f}")
print(f"  Hybrid:        {np.mean(hyb_sims):.3f} ± {np.std(hyb_sims):.3f}")
print(f"\nLower is better (more separation between emotions)")

