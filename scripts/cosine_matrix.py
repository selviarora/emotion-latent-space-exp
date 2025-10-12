import os, glob, numpy as np

# Find all embeddings like embeddings/actorXX/{happy,angry}_layer5.npy
pairs = []
for actor_dir in sorted(glob.glob("embeddings/actor*")):
    a = os.path.basename(actor_dir)
    h = os.path.join(actor_dir, "happy_layer5.npy")
    g = os.path.join(actor_dir, "angry_layer5.npy")
    if os.path.exists(h) and os.path.exists(g):
        pairs.append((f"{a}_happy", h))
        pairs.append((f"{a}_angry", g))

names = [n for n, _ in pairs]
vecs = [np.load(p).astype(np.float32) for _, p in pairs]
X = np.stack(vecs)  # shape [N, 768]

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Print cosine matrix
print("Cosine similarity matrix:")
print(" " * 12 + "  ".join([f"{n:>12s}" for n in names]))
for i, ni in enumerate(names):
    row = [f"{cos(X[i], X[j]):.3f}" for j in range(len(names))]
    print(f"{ni:12s}  " + "  ".join(row))

# Quick aggregates
actors = sorted(set(n.split('_')[0] for n in names))
def get(name): return X[names.index(name)]
emo_same, spk_same = [], []
for a in actors:
    if f"{a}_happy" in names and f"{a}_angry" in names:
        spk_same.append(cos(get(f"{a}_happy"), get(f"{a}_angry")))
for i, ai in enumerate(actors):
    for j, aj in enumerate(actors):
        if j <= i: continue
        if f"{ai}_happy" in names and f"{aj}_happy" in names:
            emo_same.append(cos(get(f"{ai}_happy"), get(f"{aj}_happy")))
        if f"{ai}_angry" in names and f"{aj}_angry" in names:
            emo_same.append(cos(get(f"{ai}_angry"), get(f"{aj}_angry")))

if emo_same:
    print(f"\nAvg cross-speaker SAME-emotion cosine: {np.mean(emo_same):.3f}")
if spk_same:
    print(f"Avg same-speaker cross-emotion cosine: {np.mean(spk_same):.3f}")
