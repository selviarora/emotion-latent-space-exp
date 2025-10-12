# scripts/extract_prosody.py

import argparse
import numpy as np
import librosa

# ---- helper: safe stats (ignores NaNs) ----
def smean(x): return float(np.nanmean(x)) if np.size(x) else 0.0
def sstd(x):  return float(np.nanstd(x))  if np.size(x) else 0.0
def srange(x):
    if np.size(x) == 0: return 0.0
    return float(np.nanmax(x) - np.nanmin(x))

def main():
    parser = argparse.ArgumentParser(description="Extract simple prosody features from a WAV.")
    parser.add_argument("--wav", required=True, help="Path to .wav file")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (we resample)")
    parser.add_argument("--out", required=True, help="Where to save .npy prosody vector")
    args = parser.parse_args()

    # 1) load audio, resample to args.sr (16 kHz default)
    #    y = mono float32 waveform; sr = actual sample rate returned
    y, sr = librosa.load(args.wav, sr=args.sr)
    # optional: trim leading/trailing silence (keeps content, removes empty tails)
    y, _ = librosa.effects.trim(y, top_db=20)

    # guard for tiny or empty audio
    if len(y) < sr * 0.2:
        raise ValueError("Audio too short after trimming (< 0.2 s)")

    # we’ll fill this dict with features, then pack to a vector
    feats = {}
    # 2) frame-wise features use a hop (default ~512 samples) → per-frame arrays
    # RMS = loudness per frame; we take std so we capture "how spiky" not absolute loudness
    rms = librosa.feature.rms(y=y)[0]  # shape 1d
    feats["rms_std"] = sstd(rms)

    # Spectral centroid ≈ brightness center of mass; we take mean and std
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # [T]
    feats["centroid_mean"] = smean(sc)
    feats["centroid_std"]  = sstd(sc)

    # 3) F0 with probabilistic YIN. Returns Hz + voiced/unvoiced as NaNs
    # choose a broad range so both male/female are covered
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    # f0 is shape [T] with NaN where unvoiced
    voiced_mask = ~np.isnan(f0)
    voiced_f0 = f0[voiced_mask]

    feats["f0_mean"]  = smean(voiced_f0)
    feats["f0_std"]   = sstd(voiced_f0)
    feats["f0_range"] = srange(voiced_f0)

    # dynamics: average absolute step change between consecutive voiced frames
    if voiced_f0.size > 1:
        delta = np.abs(np.diff(voiced_f0))
        feats["f0_delta_mean"] = smean(delta)
    else:
        feats["f0_delta_mean"] = 0.0

    # voiced fraction: percentage of frames where pitch is defined
    feats["voiced_fraction"] = float(np.mean(voiced_mask))
    # 4) pack in a fixed order → small vector (~8 dims)
    order = [
        "f0_mean", "f0_std", "f0_range", "f0_delta_mean",
        "voiced_fraction", "rms_std",
        "centroid_mean", "centroid_std",
    ]
    vec = np.array([feats[k] for k in order], dtype=np.float32)
    np.save(args.out, vec)

    # tiny print so you can see values
    print("Prosody vector ({} dims):".format(len(order)))
    for k in order:
        print(f"  {k:16s} {feats[k]:.4f}")
    print(f"[save] wrote {args.out}")

if __name__ == "__main__":
    main()
