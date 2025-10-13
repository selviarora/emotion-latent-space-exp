import numpy as np
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

FILE_PATH = "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01/03-01-05-02-02-01-01.wav"
SR = 16000  # sample rate

# Load and trim audio
wav, sr = librosa.load(FILE_PATH, sr=SR)
wav, _ = librosa.effects.trim(wav, top_db=20)
wav = wav.astype(np.float32)

print(f"[load] sr={sr} Hz duration={len(wav)/sr:.2f}s shape={wav.shape}")

# Load processor and model
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

device = torch.device("cpu")
model.to(device)

# Prepare model input
inputs = processor(wav, sampling_rate=SR, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Debug input tensor
for k, v in inputs.items():
    print(f"[inputs] {k}: shape={tuple(v.shape)} dtype={v.dtype}")

# Forward pass to extract hidden states
with torch.no_grad():
    out = model(**inputs)

print(f"[debug] hidden_states count = {len(out.hidden_states)}")  # should be 13
print(f"[debug] mid-layer shape = {tuple(out.hidden_states[6].shape)}")

# Layer 6 embedding
H = out.hidden_states[3] 
# average mid layers 5–8
#mids = [out.hidden_states[i] for i in (5,6,7,8)]  # four layers
#H = torch.stack(mids, dim=0).mean(dim=0)        # [1, T, 768]
#embedding = H.mean(dim=1).squeeze(0)
X = H.squeeze(0)                        # [T, 768]
w = torch.linalg.norm(X, dim=1) + 1e-8  # [T] per-frame “energy”
embedding = (X * w[:, None]).sum(dim=0) / w.sum()

print(f"[model] final embedding shape: {tuple(embedding.shape)}")
print("[preview] first 12 numbers:", embedding[:12].cpu().numpy().round(4).tolist())

# Save
out_path = "../embeddings/angry/actor01_layer3_energy_trim.npy"
np.save(out_path, embedding.cpu().numpy())
print(f"[save] wrote {out_path}")
