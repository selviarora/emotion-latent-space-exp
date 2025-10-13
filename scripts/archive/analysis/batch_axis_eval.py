import glob, numpy as np, argparse, torch, librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR=16000
proc=None; model=None

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
        H = out.hidden_states[layer_index].squeeze(0)  # [T,768]
        w = (H.pow(2).sum(dim=1) + 1e-8); w = w / w.sum()
        emb = (H * w.unsqueeze(1)).sum(dim=0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb)+1e-8)
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--root", required=True, help="folder that contains Actor_* dirs")
    args = ap.parse_args()

    axis = np.load(args.axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    happy_paths = sorted(glob.glob(f"{args.root}/Actor_*/03-01-03-02-02-01-*.wav"))
    angry_paths = sorted(glob.glob(f"{args.root}/Actor_*/03-01-05-02-02-01-*.wav"))

    scores = {"happy":[], "angry":[]}
    for p in happy_paths:
        e = embed_layer5_energy(p, layer_index=5)
        scores["happy"].append(float(np.dot(e, axis)))
    for p in angry_paths:
        e = embed_layer5_energy(p, layer_index=5)
        scores["angry"].append(float(np.dot(e, axis)))

    h = np.array(scores["happy"]); a = np.array(scores["angry"])
    print(f"Happy mean={h.mean():.3f} ± {h.std():.3f}  n={len(h)}")
    print(f"Angry mean={a.mean():.3f} ± {a.std():.3f}  n={len(a)}")
    # quick separability metric: Cohen's d
    d = (a.mean()-h.mean())/np.sqrt(0.5*(a.var()+h.var())+1e-8)
    print(f"Cohen's d (effect size): {d:.2f}  (0.2=small, 0.5=med, 0.8=large)")
    # hit-rate at threshold = midpoint
    thr = 0.5*(a.mean()+h.mean())
    acc = ( (a>thr).mean()*0.5 + (h<thr).mean()*0.5 )
    print(f"Midpoint-threshold accuracy (zero-shot): {acc*100:.1f}%")

if __name__ == "__main__":
    main()

