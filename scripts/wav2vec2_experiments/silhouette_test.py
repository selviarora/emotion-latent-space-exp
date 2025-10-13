import os
import glob
import numpy as np
from sklearn.metrics import silhouette_score

# === Step 1: Gather all hybrid embeddings ===
happy_files = sorted(glob.glob("embeddings/actor*/happy_hybrid.npy"))
angry_files = sorted(glob.glob("embeddings/actor*/angry_hybrid.npy"))

X = []     # embeddings
y = []     # labels: 0 = happy, 1 = angry

for file in happy_files:
    X.append(np.load(file))
    y.append(0)

for file in angry_files:
    X.append(np.load(file))
    y.append(1)

X = np.vstack(X)  # Shape: [N_samples, 776]
y = np.array(y)

print(f"[INFO] Loaded {len(X)} embeddings. Shape = {X.shape}")

# === Step 2: Compute silhouette score ===
score = silhouette_score(X, y, metric="cosine")
print(f"\n🎯 Silhouette Score (cosine distance): {score:.3f}")

# Interpretation helper
if score < 0.1:
    print("→ Very weak separation (clusters overlapping).")
elif score < 0.25:
    print("→ Some separation, but not very clean.")
elif score < 0.4:
    print("→ Moderate separation. Real signal starting to show.")
else:
    print("→ Strong structure — emotional geometry is emerging!")
