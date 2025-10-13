import glob, os
import numpy as np

def load_pairs(pattern_happy, pattern_angry, label="HYBRID"):
    happy = sorted(glob.glob(pattern_happy))
    angry = sorted(glob.glob(pattern_angry))
    actors = []

    # Expect matched counts like embeddings/actor03/happy_hybrid.npy etc.
    # We’ll map by actor dir name to pair correctly.
    h_by_actor = {os.path.dirname(p): p for p in happy}
    a_by_actor = {os.path.dirname(p): p for p in angry}
    commons = sorted(set(h_by_actor.keys()) & set(a_by_actor.keys()))

    axes = []
    for actor_dir in commons:
        h = np.load(h_by_actor[actor_dir])
        a = np.load(a_by_actor[actor_dir])
        delta = a - h                  # emotion axis for this actor
        n = np.linalg.norm(delta) + 1e-8
        v = delta / n                  # unit direction
        axes.append(v)
        actors.append(os.path.basename(actor_dir))  # e.g., actor03

    axes = np.stack(axes, axis=0) if axes else np.empty((0,0))
    print(f"[{label}] loaded {axes.shape[0]} actor axes; dim={axes.shape[1] if axes.size else 0}")
    return actors, axes

def pairwise_cosine(U):
    # U: [N, D] unit vectors
    return U @ U.T  # cosine matrix since rows are unit-normalized

def summarize_alignment(axes, label="HYBRID"):
    if axes.shape[0] < 2:
        print(f"[{label}] not enough actors for alignment.")
        return

    C = pairwise_cosine(axes)
    # off-diagonal cosines
    off_diag = C[~np.eye(C.shape[0], dtype=bool)]
    mean_align = off_diag.mean()
    std_align  = off_diag.std()

    # “Global” axis = mean of unit vectors → re-normalize
    g = axes.mean(axis=0)
    g /= (np.linalg.norm(g) + 1e-8)
    # projection of each actor’s axis onto global axis
    proj = axes @ g
    R = np.linalg.norm(axes.mean(axis=0))  # Rayleigh resultant length (0..1)

    print("\n====================================================")
    print(f"[{label}] EMOTION-AXIS ALIGNMENT")
    print("----------------------------------------------------")
    print(f"Mean pairwise cosine (off-diagonal): {mean_align:.3f} ± {std_align:.3f}")
    print(f"Global-axis projection mean:         {proj.mean():.3f} ± {proj.std():.3f}")
    print(f"Resultant length R (0=no alignment, 1=perfect): {R:.3f}")

    # Quick interpretation
    if mean_align >= 0.6 and R >= 0.6:
        verdict = "STRONG universal direction"
    elif mean_align >= 0.35 and R >= 0.35:
        verdict = "MODERATE shared direction"
    else:
        verdict = "WEAK/actor-specific directions"
    print(f"Verdict: {verdict}")

    # Optional: print a tiny cosine matrix preview
    n = min(8, C.shape[0])
    print("\nCosine alignment (first 8 actors max):")
    print(np.round(C[:n,:n], 3))

if __name__ == "__main__":
    # HYBRID (776-D) — your main result
    actors_h, axes_h = load_pairs("embeddings/actor*/happy_hybrid.npy",
                                  "embeddings/actor*/angry_hybrid.npy",
                                  label="HYBRID")
    summarize_alignment(axes_h, "HYBRID")

    # LAYER5 only (768-D) — does wav2vec alone already have direction?
    actors_l, axes_l = load_pairs("embeddings/actor*/happy_layer5.npy",
                                  "embeddings/actor*/angry_layer5.npy",
                                  label="LAYER5")
    summarize_alignment(axes_l, "LAYER5")

    # PROSODY only (8-D) — is prosody's direction stable across actors?
    actors_p, axes_p = load_pairs("embeddings/actor*/happy_prosody.npy",
                                  "embeddings/actor*/angry_prosody.npy",
                                  label="PROSODY")
    summarize_alignment(axes_p, "PROSODY")
