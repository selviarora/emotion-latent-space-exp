#!/usr/bin/env python3
"""
Emotion Slider - Prosody-Based Steering v1
Adjusts audio arousal/emotion using learned emotion axis
"""

import numpy as np
import argparse
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR = 16000

def embed_layer5_energy(wav, layer_index=5):
    """Extract layer5 embedding from audio"""
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
    
    wav = wav.astype(np.float32)
    inp = proc(wav, sampling_rate=SR, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        out = model(**inp)
        H = out.hidden_states[layer_index].squeeze(0)
        w = (H.pow(2).sum(dim=1) + 1e-8); w = w / w.sum()
        emb = (H * w.unsqueeze(1)).sum(dim=0).cpu().numpy()
    
    emb = emb / (np.linalg.norm(emb)+1e-8)
    return emb

def create_hybrid_embedding(wav):
    """Create hybrid embedding (layer5 + prosody)"""
    emb = embed_layer5_energy(wav, layer_index=5)
    
    # Prosody features
    prosody_mean = np.load('embeddings/prosody_mean.npy')
    prosody_std = np.load('embeddings/prosody_std.npy')
    
    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
    f0 = librosa.yin(wav, fmin=80, fmax=400, sr=SR)
    f0_valid = f0[f0 > 0]
    
    prosody_feats = np.array([
        np.mean(f0_valid) if len(f0_valid) > 0 else 0,
        np.std(f0_valid) if len(f0_valid) > 0 else 0,
        np.ptp(f0_valid) if len(f0_valid) > 0 else 0,
        np.mean(rms),
        np.std(rms),
        np.ptp(rms),
        len(wav) / SR,
        SR / len(wav) if len(wav) > 0 else 0
    ])
    
    prosody_feats = (prosody_feats - prosody_mean) / (prosody_std + 1e-6)
    
    return np.concatenate([emb, prosody_feats])

def apply_prosody_shift(audio, sr, pitch_shift_semitones, energy_gain_db):
    """Apply prosody changes to audio"""
    
    # Pitch shift
    if abs(pitch_shift_semitones) > 0.01:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_semitones)
    
    # Energy/loudness adjustment (in dB)
    if abs(energy_gain_db) > 0.01:
        gain_linear = 10 ** (energy_gain_db / 20)
        audio = audio * gain_linear
    
    # Normalize to prevent clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
    
    return audio

def main():
    ap = argparse.ArgumentParser(description="Emotion Slider - Adjust audio arousal/intensity")
    ap.add_argument("--input", required=True, help="Input audio file")
    ap.add_argument("--output", required=True, help="Output audio file")
    ap.add_argument("--slider", type=float, default=0.0, 
                    help="Emotion slider: -1 (calm) to +1 (angry). 0 = no change")
    ap.add_argument("--mode", choices=["subtle", "creative"], default="subtle",
                    help="subtle=layer5 (consistent), creative=hybrid (dramatic)")
    args = ap.parse_args()
    
    # Validate slider range
    if args.slider < -1 or args.slider > 1:
        print("ERROR: --slider must be between -1 and +1")
        return
    
    # Load audio
    print(f"Loading: {args.input}")
    audio, sr = librosa.load(args.input, sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Load axis
    if args.mode == "subtle":
        axis = np.load('emotion_axis_layer5_normalized.npy')
        metadata = np.load('emotion_axis_layer5_metadata.npy')
        print("Mode: SUBTLE (Layer5 - consistent, professional)")
    else:
        axis = np.load('emotion_axis_hybrid_normalized.npy')
        metadata = np.load('emotion_axis_hybrid_metadata.npy')
        print("Mode: CREATIVE (Hybrid - dramatic, expressive)")
    
    scale, midpoint = metadata
    
    # Get current emotion score
    if args.mode == "subtle":
        emb = embed_layer5_energy(audio)
    else:
        emb = create_hybrid_embedding(audio)
    
    current_score = float(np.dot(emb, axis))
    normalized_score = (current_score - midpoint) * scale
    
    print(f"\nCurrent arousal score: {normalized_score:.2f}")
    print(f"Target arousal score: {args.slider:.2f}")
    print(f"Delta: {args.slider - normalized_score:+.2f}")
    
    # Map slider change to prosody adjustments
    # Based on correlation analysis: axis correlates with pitch and energy
    
    delta = args.slider - normalized_score
    
    # Empirical mapping (tuned based on your 0.779 correlation)
    # Moving toward angry (+): increase pitch and energy
    # Moving toward calm (-): decrease pitch and energy
    
    # REDUCED for better quality (was too aggressive!)
    pitch_shift = delta * 0.5  # semitones (±0.5 semitones per full unit - more subtle)
    energy_gain = delta * 1.5  # dB (±1.5 dB per full unit - more subtle)
    
    print(f"\nApplying changes:")
    print(f"  Pitch shift: {pitch_shift:+.1f} semitones")
    print(f"  Energy gain: {energy_gain:+.1f} dB")
    
    # Apply transformation
    audio_modified = apply_prosody_shift(audio, SR, pitch_shift, energy_gain)
    
    # Save output
    sf.write(args.output, audio_modified, SR)
    print(f"\nSaved: {args.output}")
    
    # Verify new score
    if args.mode == "subtle":
        emb_new = embed_layer5_energy(audio_modified)
    else:
        emb_new = create_hybrid_embedding(audio_modified)
    
    new_score = float(np.dot(emb_new, axis))
    normalized_new = (new_score - midpoint) * scale
    
    print(f"New arousal score: {normalized_new:.2f}")
    print(f"Achieved delta: {normalized_new - normalized_score:+.2f} (target was {delta:+.2f})")
    
    print("\n" + "="*60)
    print("LISTEN TO THE OUTPUT!")
    print("Does it sound more calm/angry as expected?")
    print("="*60)

if __name__ == "__main__":
    main()

