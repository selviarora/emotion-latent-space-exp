import numpy as np

print("=== Checking prosody features ===")
for actor in ['actor01', 'actor02', 'actor03', 'actor04']:
    try:
        h_pros = np.load(f'embeddings/{actor}/happy_prosody.npy')
        a_pros = np.load(f'embeddings/{actor}/angry_prosody.npy')
        print(f"\n{actor}:")
        print(f"  happy prosody: {h_pros}")
        print(f"  angry prosody: {a_pros}")
        print(f"  different? {not np.allclose(h_pros, a_pros)}")
    except:
        print(f"\n{actor}: files not found")

print("\n\n=== Checking layer5 embeddings ===")
for actor in ['actor01', 'actor02', 'actor03', 'actor04']:
    try:
        h_l5 = np.load(f'embeddings/{actor}/happy_layer5.npy')
        a_l5 = np.load(f'embeddings/{actor}/angry_layer5.npy')
        print(f"\n{actor}:")
        print(f"  happy_layer5 shape: {h_l5.shape}, norm: {np.linalg.norm(h_l5):.3f}")
        print(f"  angry_layer5 shape: {a_l5.shape}, norm: {np.linalg.norm(a_l5):.3f}")
        print(f"  cosine similarity: {np.dot(h_l5, a_l5) / (np.linalg.norm(h_l5) * np.linalg.norm(a_l5)):.3f}")
    except Exception as e:
        print(f"\n{actor}: {e}")

print("\n\n=== Checking hybrid embeddings ===")
for actor in ['actor01', 'actor02', 'actor03', 'actor04']:
    try:
        h_hyb = np.load(f'embeddings/{actor}/happy_hybrid.npy')
        a_hyb = np.load(f'embeddings/{actor}/angry_hybrid.npy')
        print(f"\n{actor}:")
        print(f"  happy_hybrid shape: {h_hyb.shape}, norm: {np.linalg.norm(h_hyb):.3f}")
        print(f"  angry_hybrid shape: {a_hyb.shape}, norm: {np.linalg.norm(a_hyb):.3f}")
        print(f"  cosine similarity: {np.dot(h_hyb, a_hyb) / (np.linalg.norm(h_hyb) * np.linalg.norm(a_hyb)):.3f}")
        print(f"  first 5 vals: {h_hyb[:5]}")
        print(f"  last 8 vals (prosody part): {h_hyb[-8:]}")
    except Exception as e:
        print(f"\n{actor}: {e}")

print("\n\n=== Checking if hybrid = layer5 ===")
h1_l5 = np.load('embeddings/actor01/happy_layer5.npy')
h1_hyb = np.load('embeddings/actor01/happy_hybrid.npy')
print(f"actor01 happy_layer5 shape: {h1_l5.shape}")
print(f"actor01 happy_hybrid shape: {h1_hyb.shape}")
print(f"First 768 of hybrid == layer5? {np.allclose(h1_hyb[:768], h1_l5)}")

