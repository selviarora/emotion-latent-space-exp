import glob, os, numpy as np, argparse

def load_pairs(pattern_happy, pattern_angry):
    H = sorted(glob.glob(pattern_happy))
    A = sorted(glob.glob(pattern_angry))
    # pair by actor dir
    h_by_actor = {os.path.dirname(p): p for p in H}
    a_by_actor = {os.path.dirname(p): p for p in A}
    commons = sorted(set(h_by_actor) & set(a_by_actor))
    axes = []
    for actdir in commons:
        h = np.load(h_by_actor[actdir])
        a = np.load(a_by_actor[actdir])
        d = a - h
        n = np.linalg.norm(d) + 1e-8
        axes.append(d / n)
    axes = np.stack(axes, 0) if axes else np.empty((0,0))
    return axes, commons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["layer5","hybrid","prosody"], default="layer5")
    ap.add_argument("--out", default="emotion_axis.npy")
    args = ap.parse_args()

    if args.modality == "layer5":
        pat_h = "embeddings/actor*/happy_layer5.npy"
        pat_a = "embeddings/actor*/angry_layer5.npy"
    elif args.modality == "hybrid":
        pat_h = "embeddings/actor*/happy_hybrid.npy"
        pat_a = "embeddings/actor*/angry_hybrid.npy"
    else:
        pat_h = "embeddings/actor*/happy_prosody.npy"
        pat_a = "embeddings/actor*/angry_prosody.npy"

    axes, actors = load_pairs(pat_h, pat_a)
    if axes.size == 0:
        print("No pairs found.")
        return

    # average then normalize → global axis
    g = axes.mean(axis=0)
    g /= (np.linalg.norm(g) + 1e-8)

    # simple diagnostics
    pairwise = axes @ axes.T
    off = pairwise[~np.eye(pairwise.shape[0],dtype=bool)]
    print(f"Loaded {axes.shape[0]} actor axes (dim={axes.shape[1]}).")
    print(f"Mean pairwise cosine among actor axes: {off.mean():.3f} ± {off.std():.3f}")
    R = np.linalg.norm(axes.mean(axis=0))
    print(f"Resultant length R (0..1): {R:.3f}")

    np.save(args.out, g)
    print(f"[save] wrote {args.out}")

if __name__ == "__main__":
    main()

