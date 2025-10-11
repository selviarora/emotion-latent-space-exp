# === Imports (tools we need) ===============================================
import os, glob, numpy as np, torch, librosa, matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# === Config you can change ===================================================
SR = 16000  # sample rate expected by wav2vec2
# Folder with your WAVs (Actor_01 only for now):
ROOT = "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01"

# We’ll visualize two emotions: happy(03) and angry(05) in STRONG intensity (..-02-..)
EMOTIONS = {"03": "happy", "05": "angry"}

LAYER_IDX = 5     # <- we found Layer 5 gives best emotion separation
TRIM = True       # remove leading/trailing silence to focus on speech
POOL = "energy"   # "mean" or "energy" — energy emphasizes emotional frames
MAX_PER_CLASS = 20  # up to N files per class to keep the plot readable

# === Pick a reducer: try UMAP, else fallback to t-SNE ========================
reducer_name = None
try:
    import umap
    reducer_name = "umap"
except Exception:
    from sklearn.manifold import TSNE
    reducer_name = "tsne"

# === Load wav2vec2 once (feature extractor + model) ==========================
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base",
    output_hidden_states=True   # <-- this gives us all layers so we can pick LAYER_IDX
).eval()

device = torch.device("cpu")
model.to(device)

# === Helper: check a filename is the right emotion & STRONG intensity ========
def is_strong_emotion(fname, wanted_codes):
    """
    RAVDESS filename pattern: MM-SS-EE-II-SR-VI-ActorID.wav
      EE = emotion (03=happy, 05=angry)
      II = intensity (01=normal, 02=strong)
    Return class name ("happy"/"angry") or None.
    """
    base = os.path.basename(fname)
    parts = base.split("-")
    if len(parts) < 7:
        return None
    emotion = parts[2]    # "03" or "05"
    intensity = parts[3]  # "01" or "02"
    if emotion in wanted_codes and intensity == "02":
        return EMOTIONS[emotion]
    return None

# === Helper: pooling over time to get one vector per clip ====================
def energy_pool(H):
    """
    H: tensor [1, T, 768] from a chosen layer.
    We compute per-frame magnitude (||frame||) and use it as weights,
    so high-activation (emotionally salient) frames count more.
    Returns: tensor [768]
    """
    X = H.squeeze(0)                           # [T, 768]
    w = torch.linalg.norm(X, dim=1) + 1e-8     # [T] non-negative weights
    return (X * w[:, None]).sum(dim=0) / w.sum()

# === Helper: full pipeline to get one embedding from a file ==================
def embed_file(path):
    """
    Load WAV -> (optional) trim -> to model inputs -> forward pass ->
    pick hidden_states[LAYER_IDX] -> pool over time -> return [768] numpy vector.
    """
    wav, sr = librosa.load(path, sr=SR)
    if TRIM:
        wav, _ = librosa.effects.trim(wav, top_db=20)
    wav = wav.astype(np.float32)

    inputs = processor(wav, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)                  # forward pass; has .hidden_states

    H = out.hidden_states[LAYER_IDX]           # choose the layer we want: [1, T, 768]

    if POOL == "mean":
        emb = H.mean(dim=1).squeeze(0)        # simple average over time -> [768]
    else:
        emb = energy_pool(H)                   # energy-weighted average -> [768]

    return emb.cpu().numpy()

# === Collect a small, balanced set of files per class ========================
files = sorted(glob.glob(os.path.join(ROOT, "*.wav")))
by_class = {"happy": [], "angry": []}
for f in files:
    cls = is_strong_emotion(f, wanted_codes=set(EMOTIONS.keys()))
    if cls is None:
        continue
    if len(by_class[cls]) < MAX_PER_CLASS:     # cap to keep plot neat
        by_class[cls].append(f)

print("[info] collected:", {k: len(v) for k, v in by_class.items()})
assert sum(len(v) for v in by_class.values()) > 1, "Not enough files found."

# === Turn each file into one 768-D embedding =================================
X_list, y_list = [], []
for cls, paths in by_class.items():
    for p in paths:
        try:
            emb = embed_file(p)                # numpy [768]
            X_list.append(emb)
            y_list.append(cls)
        except Exception as e:
            print("[warn] failed:", p, e)

X = np.stack(X_list, axis=0)   # shape [N, 768], N = number of clips
y = np.array(y_list)

# === Dimensionality reduction: 768D -> 2D for plotting =======================
if reducer_name == "umap":
    reducer = umap.UMAP(
        n_neighbors=10,      # how local vs global; smaller → tighter local clusters
        min_dist=0.1,        # how packed points can be; smaller → tighter
        metric="cosine",     # cosine works well for embeddings
        random_state=42
    )
    Z = reducer.fit_transform(X)  # Z shape: [N, 2]
    title = "UMAP (cosine) — Layer 5, energy pooling"
else:
    from sklearn.manifold import TSNE
    reducer = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(X)//4)),  # auto-ish setting based on N
        init="pca",
        metric="cosine",
        random_state=42
    )
    Z = reducer.fit_transform(X)
    title = "t-SNE (cosine) — Layer 5, energy pooling"

# === Plot: each clip becomes a dot; color by emotion =========================
# --- scatter plot with centroids + separation line ---
colors = {"happy": "#1f77b4", "angry": "#d62728"}  # blue/red
plt.figure(figsize=(6.5, 5.2))

# scatter points
for cls in ["happy", "angry"]:
    mask = (y == cls)
    plt.scatter(Z[mask, 0], Z[mask, 1], s=34, alpha=0.85, c=colors[cls], label=cls, edgecolors="none")

# compute centroids in 2D
mask_h = (y == "happy")
mask_a = (y == "angry")
c_h = Z[mask_h].mean(axis=0)   # [x, y]
c_a = Z[mask_a].mean(axis=0)   # [x, y]

# plot centroids as big stars
plt.scatter([c_h[0]], [c_h[1]], marker="*", s=220, c=colors["happy"], edgecolors="k", linewidths=0.7, label="happy centroid")
plt.scatter([c_a[0]], [c_a[1]], marker="*", s=220, c=colors["angry"], edgecolors="k", linewidths=0.7, label="angry centroid")

# draw faint line connecting the two centroids (purely for illustration)
plt.plot([c_h[0], c_a[0]], [c_h[1], c_a[1]], linestyle="--", linewidth=1.0, color="#666666", alpha=0.6)

# perpendicular-bisector as a simple separation boundary:
m = (c_h + c_a) / 2.0           # midpoint
d = c_a - c_h                   # direction from happy -> angry
n = np.array([-d[1], d[0]])     # perpendicular to d

# create an x-range spanning the plot, then compute y from the line equation n·(x - m)=0
xmin, xmax = Z[:,0].min() - 0.5, Z[:,0].max() + 0.5
if abs(n[1]) > 1e-8:
    xs = np.array([xmin, xmax])
    ys = m[1] - (n[0]/n[1]) * (xs - m[0])
    plt.plot(xs, ys, color="black", linewidth=1.5, alpha=0.8, label="perp. bisector")
else:
    # vertical line if n[1] ~ 0
    plt.axvline(m[0], color="black", linewidth=1.5, alpha=0.8, label="perp. bisector")

plt.title(title)
plt.xlabel("dim-1"); plt.ylabel("dim-2")
plt.legend(loc="best", frameon=True)
plt.tight_layout()
plt.savefig("../outputs/l5_emotion_umap_centroids.png", dpi=180)
print("[save] wrote ../outputs/l5_emotion_umap_centroids.png")
