import glob, numpy as np, argparse, torch, librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR=16000
proc=None; model=None

# RAVDESS emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm", 
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def embed_layer5_energy(wav_path, layer_index=5):
    global proc, model
    if proc is None:
        proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    if model is None:
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

    wav, sr = librosa.load(wav_path, sr=SR)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    wav = wav.astype(np.float32)

    inp = proc(wav, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inp)
        H = out.hidden_states[layer_index].squeeze(0)
        w = (H.pow(2).sum(dim=1) + 1e-8); w = w / w.sum()
        emb = (H * w.unsqueeze(1)).sum(dim=0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb)+1e-8)
    return emb

def extract_emotion(filename):
    # RAVDESS format: 03-01-XX-... where XX is emotion code
    parts = filename.split('/')[-1].split('-')
    if len(parts) >= 3:
        return EMOTION_MAP.get(parts[2], "unknown")
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, help="emotion_axis_*.npy")
    ap.add_argument("--root", required=True, help="folder containing Actor_* dirs")
    ap.add_argument("--emotions", nargs="+", default=["happy", "angry", "sad", "calm", "fearful"],
                    help="Emotions to test")
    args = ap.parse_args()

    axis = np.load(args.axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # Collect all wav files
    all_wavs = glob.glob(f"{args.root}/Actor_*/03-01-*-02-02-01-*.wav")
    
    # Group by emotion
    scores_by_emotion = {emo: [] for emo in args.emotions}
    
    print(f"Processing {len(all_wavs)} files...")
    for wav_path in all_wavs:
        emotion = extract_emotion(wav_path)
        if emotion in scores_by_emotion:
            emb = embed_layer5_energy(wav_path, layer_index=5)
            score = float(np.dot(emb, axis))
            scores_by_emotion[emotion].append(score)

    # Print results
    print("\n" + "="*60)
    print(f"AXIS PROJECTION SCORES (axis built from angry - happy)")
    print("-"*60)
    
    results = []
    for emotion in args.emotions:
        scores = np.array(scores_by_emotion[emotion])
        if len(scores) > 0:
            results.append({
                'emotion': emotion,
                'mean': scores.mean(),
                'std': scores.std(),
                'n': len(scores)
            })
            print(f"{emotion:12s}: {scores.mean():+.3f} ± {scores.std():.3f}  (n={len(scores)})")
        else:
            print(f"{emotion:12s}: no samples found")
    
    # Pairwise comparisons
    print("\n" + "="*60)
    print("PAIRWISE SEPARABILITY (Cohen's d)")
    print("-"*60)
    
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            s1 = np.array(scores_by_emotion[r1['emotion']])
            s2 = np.array(scores_by_emotion[r2['emotion']])
            d = (s2.mean() - s1.mean()) / np.sqrt(0.5*(s1.var() + s2.var()) + 1e-8)
            print(f"{r1['emotion']:10s} vs {r2['emotion']:10s}: d={d:+.2f}  (diff={s2.mean()-s1.mean():+.3f})")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("- Positive scores = closer to 'angry' side")
    print("- Negative scores = closer to 'happy' side")
    print("- If sad/calm are negative → axis captures valence (positive/negative emotion)")
    print("- If fearful is positive → axis might capture arousal (high/low energy)")
    print("="*60)

if __name__ == "__main__":
    main()

