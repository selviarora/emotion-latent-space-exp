#!/usr/bin/env python3
"""
Emotion Slider using Coqui TTS voice cloning + wav2vec2 validation
"""
import argparse
import numpy as np
import torch
import librosa
from TTS.api import TTS
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

WAV2VEC_SR = 16000

def extract_wav2vec_score(audio_path, axis):
    """Score an audio file using the emotion axis."""
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()
    
    wav, _ = librosa.load(audio_path, sr=WAV2VEC_SR)
    inp = proc(wav, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        out = model(**inp)
        H = out.hidden_states[5].squeeze(0)
        energy = H.pow(2).sum(dim=1) + 1e-8
        weights = energy / energy.sum()
        emb = (H * weights.unsqueeze(1)).sum(dim=0).cpu().numpy()
    
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    score = float(np.dot(emb, axis))
    return score

def main():
    ap = argparse.ArgumentParser(description="Emotion Slider with TTS Voice Cloning")
    ap.add_argument("--text", required=True, help="Text to speak")
    ap.add_argument("--slider", type=float, default=0.0, 
                    help="Emotion slider: -1.0 (calm) to +1.0 (angry)")
    ap.add_argument("--calm_ref", required=True, help="Path to calm reference audio")
    ap.add_argument("--angry_ref", required=True, help="Path to angry reference audio")
    ap.add_argument("--output", default="output_emotion.wav")
    args = ap.parse_args()
    
    print("🎙️  Emotion Slider with Voice Cloning")
    print("="*60)
    
    # Load emotion axis
    axis = np.load('models/emotion_axis_layer5.npy')
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    print(f"✓ Loaded emotion axis")
    
    # Score references
    print(f"\n📊 Scoring reference clips:")
    calm_score = extract_wav2vec_score(args.calm_ref, axis)
    angry_score = extract_wav2vec_score(args.angry_ref, axis)
    print(f"   Calm reference:  {calm_score:+.4f}")
    print(f"   Angry reference: {angry_score:+.4f}")
    
    # Initialize TTS (using YourTTS for voice cloning)
    print(f"\n🔧 Loading TTS model...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
    
    # Map slider to reference weights
    # slider: -1.0 = full calm, 0.0 = neutral, +1.0 = full angry
    slider_normalized = (args.slider + 1.0) / 2.0  # Map to [0, 1]
    
    # Choose primary reference based on slider
    if slider_normalized < 0.5:
        # Use calm as speaker, adjust towards neutral
        speaker_wav = args.calm_ref
        print(f"\n🎚️  Slider: {args.slider:.2f} → Using CALM voice")
    else:
        # Use angry as speaker
        speaker_wav = args.angry_ref
        print(f"\n🎚️  Slider: {args.slider:.2f} → Using ANGRY voice")
    
    # Generate speech
    print(f"🔊 Synthesizing: \"{args.text}\"...")
    tts.tts_to_file(
        text=args.text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=args.output
    )
    
    # Validate output with axis
    print(f"\n✓ Saved: {args.output}")
    output_score = extract_wav2vec_score(args.output, axis)
    print(f"\n📈 Validation:")
    print(f"   Output axis score: {output_score:+.4f}")
    print(f"   Target (slider {args.slider:+.2f}): between {calm_score:+.4f} and {angry_score:+.4f}")
    
    # Success check
    in_range = min(calm_score, angry_score) <= output_score <= max(calm_score, angry_score)
    if in_range:
        print(f"   ✅ Output emotion is in expected range!")
    else:
        print(f"   ⚠️  Output emotion outside expected range (but might still sound good)")
    
    print("\n" + "="*60)
    print("Done! Listen to the output to verify emotion.")

if __name__ == "__main__":
    main()

