# scripts/make_hybrid.py
import argparse, numpy as np, pathlib

def main():
    ap = argparse.ArgumentParser(description="Concatenate wav2vec2 embedding (768D) with prosody (8D).")
    ap.add_argument("--embed",   required=True, help="Path to wav2vec2 .npy (e.g., *_layer5.npy)")
    ap.add_argument("--prosody", required=True, help="Path to prosody .npy (8 dims)")
    ap.add_argument("--out",     required=True, help="Output path for 776D hybrid .npy")
    ap.add_argument("--prosody_mean", help="(Optional) path to npy of prosody mean (8,). If given, z-score prosody.")
    ap.add_argument("--prosody_std",  help="(Optional) path to npy of prosody std  (8,). If given, z-score prosody.")
    args = ap.parse_args()

    e = np.load(args.embed)       # (768,)
    p = np.load(args.prosody)     # (8,)

    if args.prosody_mean and args.prosody_std:
        mu = np.load(args.prosody_mean)  # (8,)
        sd = np.load(args.prosody_std)   # (8,)
        eps = 1e-6
        p = (p - mu) / (sd + eps)
    else:
        # If no normalization params provided, at least scale prosody to similar magnitude as embeddings
        # Typical wav2vec2 embedding values are ~0.1, so scale prosody to similar range
        p = p / (np.linalg.norm(p) + 1e-9)  # normalize to unit norm
        p = p * np.linalg.norm(e) / np.sqrt(768) * np.sqrt(8)  # scale to similar per-dim magnitude

    h = np.concatenate([e, p], axis=0)   # (776,)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, h)
    print(f"[shapes] embed={e.shape} prosody={p.shape} hybrid={h.shape}")
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
