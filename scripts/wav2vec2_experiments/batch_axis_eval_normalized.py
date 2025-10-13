import glob, numpy as np, argparse, torch, librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from collections import defaultdict

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

def extract_actor_id(path):
    # Extract actor number from path like "Actor_01" or "actor01"
    import re
    match = re.search(r'[Aa]ctor[_-]?(\d+)', path)
    return match.group(1) if match else "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--root", required=True, help="folder that contains Actor_* dirs")
    ap.add_argument("--center_on_happy", action="store_true", help="Subtract happy centroid before projection")
    args = ap.parse_args()

    axis = np.load(args.axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    happy_paths = sorted(glob.glob(f"{args.root}/Actor_*/03-01-03-02-02-01-*.wav"))
    angry_paths = sorted(glob.glob(f"{args.root}/Actor_*/03-01-05-02-02-01-*.wav"))

    # Collect embeddings AND track by actor
    happy_by_actor = defaultdict(list)
    angry_by_actor = defaultdict(list)
    
    print("Processing happy files...")
    for p in happy_paths:
        actor_id = extract_actor_id(p)
        e = embed_layer5_energy(p, layer_index=5)
        happy_by_actor[actor_id].append(e)
    
    print("Processing angry files...")
    for p in angry_paths:
        actor_id = extract_actor_id(p)
        e = embed_layer5_energy(p, layer_index=5)
        angry_by_actor[actor_id].append(e)

    # Optional: Center on happy
    happy_centroid = None
    if args.center_on_happy:
        all_happy = np.vstack([np.array(v) for v in happy_by_actor.values()])
        happy_centroid = all_happy.mean(axis=0)
        print(f"[Centering] Happy centroid computed from {len(all_happy)} samples")

    # Compute raw scores
    raw_happy = []
    raw_angry = []
    
    for actor_id in happy_by_actor:
        for e in happy_by_actor[actor_id]:
            e_centered = e - happy_centroid if happy_centroid is not None else e
            raw_happy.append(float(np.dot(e_centered, axis)))
    
    for actor_id in angry_by_actor:
        for e in angry_by_actor[actor_id]:
            e_centered = e - happy_centroid if happy_centroid is not None else e
            raw_angry.append(float(np.dot(e_centered, axis)))

    # Per-actor normalization
    norm_happy = []
    norm_angry = []
    
    for actor_id in set(happy_by_actor.keys()) | set(angry_by_actor.keys()):
        actor_scores = []
        
        # Collect all scores for this actor
        for e in happy_by_actor[actor_id]:
            e_centered = e - happy_centroid if happy_centroid is not None else e
            actor_scores.append(float(np.dot(e_centered, axis)))
        for e in angry_by_actor[actor_id]:
            e_centered = e - happy_centroid if happy_centroid is not None else e
            actor_scores.append(float(np.dot(e_centered, axis)))
        
        # Normalize (z-score within actor)
        if len(actor_scores) > 0:
            mean_a = np.mean(actor_scores)
            std_a = np.std(actor_scores) + 1e-8
            
            for e in happy_by_actor[actor_id]:
                e_centered = e - happy_centroid if happy_centroid is not None else e
                score = float(np.dot(e_centered, axis))
                norm_happy.append((score - mean_a) / std_a)
            
            for e in angry_by_actor[actor_id]:
                e_centered = e - happy_centroid if happy_centroid is not None else e
                score = float(np.dot(e_centered, axis))
                norm_angry.append((score - mean_a) / std_a)

    # Convert to arrays
    raw_h = np.array(raw_happy)
    raw_a = np.array(raw_angry)
    norm_h = np.array(norm_happy)
    norm_a = np.array(norm_angry)

    # Print results
    print("\n" + "="*60)
    print("RAW SCORES (no normalization)")
    print("-"*60)
    print(f"Happy mean={raw_h.mean():.3f} ± {raw_h.std():.3f}  n={len(raw_h)}")
    print(f"Angry mean={raw_a.mean():.3f} ± {raw_a.std():.3f}  n={len(raw_a)}")
    d_raw = (raw_a.mean()-raw_h.mean())/np.sqrt(0.5*(raw_a.var()+raw_h.var())+1e-8)
    print(f"Cohen's d: {d_raw:.2f}")
    thr_raw = 0.5*(raw_a.mean()+raw_h.mean())
    acc_raw = ( (raw_a>thr_raw).mean()*0.5 + (raw_h<thr_raw).mean()*0.5 )
    print(f"Midpoint accuracy: {acc_raw*100:.1f}%")

    print("\n" + "="*60)
    print("NORMALIZED SCORES (per-actor z-score)")
    print("-"*60)
    print(f"Happy mean={norm_h.mean():.3f} ± {norm_h.std():.3f}  n={len(norm_h)}")
    print(f"Angry mean={norm_a.mean():.3f} ± {norm_a.std():.3f}  n={len(norm_a)}")
    d_norm = (norm_a.mean()-norm_h.mean())/np.sqrt(0.5*(norm_a.var()+norm_h.var())+1e-8)
    print(f"Cohen's d: {d_norm:.2f}")
    thr_norm = 0.5*(norm_a.mean()+norm_h.mean())
    acc_norm = ( (norm_a>thr_norm).mean()*0.5 + (norm_h<thr_norm).mean()*0.5 )
    print(f"Midpoint accuracy: {acc_norm*100:.1f}%")
    
    print("\n" + "="*60)
    print(f"IMPROVEMENT: Accuracy {acc_raw*100:.1f}% → {acc_norm*100:.1f}% ({(acc_norm-acc_raw)*100:+.1f}%)")
    print(f"             Cohen's d {d_raw:.2f} → {d_norm:.2f} ({d_norm-d_raw:+.2f})")
    print("="*60)

if __name__ == "__main__":
    main()

