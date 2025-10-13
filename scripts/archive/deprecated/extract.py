# scripts/extract.py
# Purpose: Extract a single 768-D clip embedding from wav2vec2 for any .wav
# Usage example:
#   python scripts/extract.py \
#       --wav "/path/Actor_02/03-01-05-02-02-01-02.wav" \
#       --out embeddings/actor02/angry_layer5.npy \
#       --layer_index 5 \
#       --pool energy

import argparse
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Wav2Vec2 expects 16 kHz mono audio
SR = 16000

def mean_pooling(H: torch.Tensor) -> torch.Tensor:
    """
    H: [1, T, 768] per-frame embeddings from a chosen hidden_states layer.
    Returns: [768] clip-level embedding by simple average across time.
    """
    return H.mean(dim=1).squeeze(0)

def energy_pooling(H: torch.Tensor) -> torch.Tensor:
    """
    Emphasize frames with higher activation energy.
    1) frame energy = sqrt(sum(feature^2))
    2) normalize weights so they sum to 1
    3) weighted sum across time
    """
    energy = (H.pow(2).sum(dim=-1) + 1e-8).sqrt()      # [1, T]
    weights = energy / energy.sum(dim=1, keepdim=True) # [1, T]
    pooled = (H * weights.unsqueeze(-1)).sum(dim=1)    # [1, 768]
    return pooled.squeeze(0)

def main():
    ap = argparse.ArgumentParser(description="Extract a 768-D embedding from wav2vec2.")
    ap.add_argument("--wav", required=True, help="Path to input .wav (will be resampled to 16 kHz)")
    ap.add_argument("--out", required=True, help="Where to save the .npy (embedding)")
    # Layer numbering note (Hugging Face hidden_states):
    #   hidden_states[0] = conv front-end output (not a transformer block)
    #   hidden_states[1] = after Transformer layer 0
    #   hidden_states[2] = after Transformer layer 1
    #   ...
    # If your best empirical layer was what you called “Layer 5” earlier AND you indexed [5] in code,
    # keep using --layer_index 5 for consistency.
    ap.add_argument("--layer_index", type=int, default=5, help="Index into hidden_states (default 5)")
    ap.add_argument("--pool", choices=["mean", "energy"], default="energy",
                    help="Pooling over time frames → single vector (default: energy)")
    ap.add_argument("--trim_db", type=float, default=20.0,
                    help="librosa trim threshold in dB (default 20)")
    args = ap.parse_args()

    # 1) Load & trim audio (mono float32 @ 16kHz)
    y, sr = librosa.load(args.wav, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=args.trim_db)
    y = y.astype(np.float32)
    print(f"[load] {args.wav}\n       sr={sr}Hz, duration={len(y)/sr:.2f}s, samples={len(y)}")

    # 2) Feature extractor + model (inference mode)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base",
                                          output_hidden_states=True).eval()

    device = torch.device("cpu")  # swap to "cuda" if you have a GPU later
    model.to(device)

    # 3) Pack waveform → model inputs (PyTorch tensors)
    inputs = processor(y, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"[inputs] input_values shape={tuple(inputs['input_values'].shape)}  dtype={inputs['input_values'].dtype}")

    # 4) Forward pass (no gradients; we’re not training)
    with torch.no_grad():
        out = model(**inputs)  # has .last_hidden_state and .hidden_states
        H = out.hidden_states[args.layer_index]  # [1, T, 768]
        print(f"[model] chosen layer index={args.layer_index}, per-frame shape={tuple(H.shape)}")

        if args.pool == "mean":
            emb = mean_pooling(H)     # [768]
        else:
            emb = energy_pooling(H)   # [768]

    # 5) Save
    np.save(args.out, emb.cpu().numpy())
    print(f"[save] {args.out}  shape={tuple(emb.shape)}  pool={args.pool}")

if __name__ == "__main__":
    main()
