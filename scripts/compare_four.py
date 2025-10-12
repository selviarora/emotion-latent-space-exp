# scripts/compare_four.py
import numpy as np

paths = {
  "A1_happy": "embeddings/actor01/happy_layer5.npy",
  "A1_angry": "embeddings/actor01/angry_layer5.npy",
  "A2_happy": "embeddings/actor02/happy_layer5.npy",
  "A2_angry": "embeddings/actor02/angry_layer5.npy",
}
names = list(paths.keys())
vecs = [np.load(paths[n]).astype(np.float32) for n in names]

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

print("Cosine similarity matrix:")
for i, ni in enumerate(names):
    row = [f"{cos(vecs[i], vecs[j]):.3f}" for j in range(len(names))]
    print(f"{ni:10s}  " + "  ".join(row))
