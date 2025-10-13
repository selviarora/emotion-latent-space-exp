#!/usr/bin/env python3
"""
Emotion Slider TTS (reference-style blending + wav2vec2 scoring)

Usage (example):
  python tts.py \
    --text "Dogs are sitting by the door." \
    --ref_calm path/to/calm_ref.wav \
    --ref_angry path/to/angry_ref.wav \
    --axis models/emotion_axis_layer5_normalized.npy \
    --out out.wav \
    --alpha 0.4

Alpha ∈ [-1, +1]:
  -1 → fully calm
   0 → 50/50 blend
  +1 → fully angry
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch

# wav2vec2 (for scoring)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# --- backend: Coqui TTS ---
# We try XTTS-v2 first (best ref-style support), then fall back to generic API.
try:
    from TTS.api import TTS as COQUI_TTS
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False


# -----------------------------
# wav2vec2 scoring (your axis)
# -----------------------------
class EmotionScorer:
    def __init__(self, axis_path: str, sr=16000, device="cpu"):
        self.sr = sr
        self.device = torch.device(device)
        self.axis = torch.tensor(
            np.load(axis_path).astype(np.float32), device=self.device
        )
        # Normalize just in case
        self.axis = self.axis / (self.axis.norm() + 1e-8)

        self.proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base", output_hidden_states=True
        ).eval().to(self.device)

    @torch.no_grad()
    def clip_embedding(self, wav: np.ndarray) -> torch.Tensor:
        # pack to tensors expected by wav2vec2
        inp = self.proc(
            wav.astype(np.float32), sampling_rate=self.sr, return_tensors="pt", padding=True
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        H = out.hidden_states[5].squeeze(0)  # layer 5, shape [T, 768]
        # energy pooling
        energy = (H**2).sum(dim=1)
        weights = energy / (energy.sum() + 1e-8)
        emb = (H * weights.unsqueeze(1)).sum(dim=0)  # [768]
        return emb

    @torch.no_grad()
    def score(self, wav: np.ndarray) -> float:
        emb = self.clip_embedding(wav)
        return float(emb @ self.axis)


# -----------------------------
# TTS backend
# -----------------------------
class TTSBackend:
    """
    Synthesizes speech given text + a single reference wav.
    We'll call it twice (calm and angry), then blend in the waveform domain.
    """

    def __init__(self, device="cpu"):
        if not COQUI_AVAILABLE:
            raise RuntimeError(
                "Coqui TTS not installed. Run: pip install TTS"
            )
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # Prefer XTTS v2 if present; otherwise use a generic English model.
        # You can change model names here if you have a local checkpoint.
        preferred = [
            "tts_models/multilingual/multi-dataset/xtts_v2",   # reference style wav support
            "tts_models/en/vctk/vits",                         # fallback multi-speaker
            "tts_models/en/ljspeech/tacotron2-DDC_ph"          # simple single-spk fallback
        ]

        last_err = None
        self.model = None
        for name in preferred:
            try:
                self.model = COQUI_TTS(name)
                break
            except Exception as e:
                last_err = e
                continue

        if self.model is None:
            raise RuntimeError(f"Could not load any TTS model. Last error: {last_err}")

        print(f"[TTS] Loaded model: {self.model.synthesizer.model_name}")

        # Some models expose language/speaker/style args; we'll pass only those they accept.
        self.is_xtts = "xtts_v2" in self.model.synthesizer.model_name

    def synth(self, text: str, ref_wav: str, sr_out=16000) -> np.ndarray:
        """
        Return mono float32 audio at sr_out.
        For XTTS we pass 'speaker_wav' (style cloning). For others, we pass what is supported.
        """
        if self.is_xtts:
            # XTTS takes a reference voice/style wav
            y = self.model.tts(
                text=text,
                speaker_wav=ref_wav,
                language="en"
            )
        else:
            # Generic path (many models ignore speaker_wav). Still synth to have a signal.
            # Some VITS models accept speaker_idx / speaker_name if multi-speaker.
            try:
                y = self.model.tts(text=text)
            except TypeError:
                # Some expect kwargs slightly differently
                y = self.model.tts(text)

        y = np.asarray(y, dtype=np.float32)
        # Resample (Coqui models often output 22.05k or 24k)
        if sr_out is not None:
            y = librosa.resample(y, orig_sr=self.model.synthesizer.output_sample_rate, target_sr=sr_out)
        # Normalize politely
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.95
        return y


# -----------------------------
# Utility: blend two audios
# -----------------------------
def time_align_and_blend(y_calm: np.ndarray, y_angry: np.ndarray, alpha: float) -> np.ndarray:
    """
    α ∈ [-1, +1]; weights:
        w_calm  = (1 - alpha) / 2
        w_angry = (1 + alpha) / 2
    Length-match by padding/truncation, then linear blend.
    """
    w_c = float((1 - alpha) * 0.5)
    w_a = float((1 + alpha) * 0.5)

    L = max(len(y_calm), len(y_angry))
    yc = np.zeros(L, dtype=np.float32)
    ya = np.zeros(L, dtype=np.float32)
    yc[: len(y_calm)] = y_calm
    ya[: len(y_angry)] = y_angry

    y = w_c * yc + w_a * ya
    # light de-click: short fade in/out
    fade = min(256, len(y)//20)
    if fade > 0:
        y[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    # renorm
    y = y / (np.max(np.abs(y)) + 1e-8) * 0.95
    return y


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Emotion Slider TTS (reference-style blend + wav2vec2 score)")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--ref_calm", required=True, help="Reference wav for calm style")
    ap.add_argument("--ref_angry", required=True, help="Reference wav for angry style")
    ap.add_argument("--axis", required=True, help="Path to emotion axis (e.g., emotion_axis_layer5_normalized.npy)")
    ap.add_argument("--out", required=True, help="Output wav path")
    ap.add_argument("--alpha", type=float, default=0.0, help="Slider [-1..+1]: -1 calm, +1 angry")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    if not COQUI_AVAILABLE:
        print("ERROR: Coqui TTS not installed. Run: pip install TTS", file=sys.stderr)
        sys.exit(1)

    # Clamp alpha
    alpha = float(np.clip(args.alpha, -1.0, 1.0))

    # Build TTS backend
    tts = TTSBackend(device=args.device)

    # Synthesize endpoints
    print("[TTS] Synth calm...")
    y_calm = tts.synth(args.text, args.ref_calm, sr_out=16000)
    print("[TTS] Synth angry...")
    y_angry = tts.synth(args.text, args.ref_angry, sr_out=16000)

    # Blend by slider
    print(f"[BLEND] alpha={alpha:+.2f}  (−1=calm … +1=angry)")
    y = time_align_and_blend(y_calm, y_angry, alpha)

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sf.write(args.out, y, 16000)
    print(f"[SAVE] {args.out}")

    # Score with wav2vec2 axis
    scorer = EmotionScorer(axis_path=args.axis, sr=16000, device=args.device)
    s_calm = scorer.score(y_calm)
    s_angr = scorer.score(y_angry)
    s_out  = scorer.score(y)

    print("\n=== wav2vec2 axis scores (higher → angrier, lower → calmer) ===")
    print(f"calm_ref : {s_calm:+.3f}")
    print(f"angry_ref: {s_angr:+.3f}")
    print(f"output   : {s_out:+.3f}   <-- should move toward angry as alpha → +1")
    print("================================================================")


if __name__ == "__main__":
    main()

