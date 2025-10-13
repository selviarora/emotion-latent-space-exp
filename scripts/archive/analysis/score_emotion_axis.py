import argparse, numpy as np, torch, librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR = 16000

def embed_layer5_energy(wav_path, layer_index=5):
    wav, sr = librosa.load(wav_path, sr=SR)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    wav = wav.astype(np.float32)

    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

    inp = proc(wav, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inp)
        H = out.hidden_states[layer_index].squeeze(0)    # [T,768]
        # energy pooling weights
        w = (H.pow(2).sum(dim=1) + 1e-8)
        w = w / w.sum()
        emb = (H * w.unsqueeze(1)).sum(dim=0).cpu().numpy()  # [768]
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--axis", required=True, help="emotion_axis_*.npy")
    args = ap.parse_args()

    axis = np.load(args.axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    emb = embed_layer5_energy(args.wav, layer_index=5 if axis.shape[0]==768 else 5)  # using layer5 embeddings
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    score = float(np.dot(emb, axis))
    print(f"Projection score along axis: {score:.3f}")
    print("Interpretation: more positive → closer to 'angry' centroid side (since axis = angry - happy).")

if __name__ == "__main__":
    main()

