import glob, numpy as np, argparse, torch, librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import matplotlib.pyplot as plt
from collections import defaultdict

SR=16000
proc=None; model=None

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

EMOTION_COLORS = {
    "calm": "#4A90E2",     # Blue - low arousal
    "sad": "#7B68EE",      # Purple - low arousal
    "happy": "#FFD700",    # Gold - mid arousal
    "angry": "#FF4444",    # Red - high arousal
    "fearful": "#FF8C00",  # Orange - high arousal
    "neutral": "#999999",
    "disgust": "#8B4513",
    "surprised": "#FF69B4"
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

def extract_prosody(wav_path):
    """Extract RMS_std and pitch_range as arousal proxies"""
    wav, sr = librosa.load(wav_path, sr=SR)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    
    # RMS energy in frames
    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
    rms_std = np.std(rms)  # Variation in energy
    
    # Pitch tracking
    f0 = librosa.yin(wav, fmin=80, fmax=400, sr=sr)
    f0_valid = f0[f0 > 0]  # Remove unvoiced frames
    pitch_range = np.ptp(f0_valid) if len(f0_valid) > 0 else 0  # Peak-to-peak
    
    return rms_std, pitch_range

def extract_emotion(filename):
    parts = filename.split('/')[-1].split('-')
    if len(parts) >= 3:
        return EMOTION_MAP.get(parts[2], "unknown")
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--arousal_feature", choices=["rms_std", "pitch_range", "both"], default="rms_std")
    ap.add_argument("--emotions", nargs="+", default=["happy", "angry", "sad", "calm", "fearful"])
    ap.add_argument("--out", default="outputs/arousal_geometry_validation.png")
    args = ap.parse_args()

    axis = np.load(args.axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Detect if hybrid axis (776-D) or layer5 only (768-D)
    is_hybrid = (axis.shape[0] == 776)
    if is_hybrid:
        print("Detected HYBRID axis (776-D) - will extract layer5 + prosody")
        prosody_mean = np.load('embeddings/stats/prosody_mean.npy')
        prosody_std = np.load('embeddings/stats/prosody_std.npy')
    else:
        print("Detected LAYER5 axis (768-D) - will extract layer5 only")

    all_wavs = glob.glob(f"{args.root}/Actor_*/03-01-*-02-02-01-*.wav")
    
    data = defaultdict(lambda: {"proj": [], "rms_std": [], "pitch_range": []})
    
    print(f"Processing {len(all_wavs)} files...")
    for wav_path in all_wavs:
        emotion = extract_emotion(wav_path)
        if emotion not in args.emotions:
            continue
        
        # Projection score
        emb = embed_layer5_energy(wav_path, layer_index=5)
        
        # If hybrid, add prosody features
        if is_hybrid:
            rms_std, pitch_range = extract_prosody(wav_path)
            # Extract full prosody (need to recompute properly)
            wav, sr = librosa.load(wav_path, sr=SR)
            wav, _ = librosa.effects.trim(wav, top_db=20)
            
            # Extract same 8 prosody features as in batch_extract_prosody.py
            rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
            f0 = librosa.yin(wav, fmin=80, fmax=400, sr=sr)
            f0_valid = f0[f0 > 0]
            
            prosody_feats = np.array([
                np.mean(f0_valid) if len(f0_valid) > 0 else 0,
                np.std(f0_valid) if len(f0_valid) > 0 else 0,
                np.ptp(f0_valid) if len(f0_valid) > 0 else 0,
                np.mean(rms),
                np.std(rms),
                np.ptp(rms),
                len(wav) / sr,  # duration
                sr / len(wav) if len(wav) > 0 else 0  # speaking rate proxy
            ])
            
            # Normalize prosody
            prosody_feats = (prosody_feats - prosody_mean) / (prosody_std + 1e-6)
            
            # Concatenate
            emb = np.concatenate([emb, prosody_feats])
        
        proj_score = float(np.dot(emb, axis))
        
        # Prosody features
        rms_std, pitch_range = extract_prosody(wav_path)
        
        data[emotion]["proj"].append(proj_score)
        data[emotion]["rms_std"].append(rms_std)
        data[emotion]["pitch_range"].append(pitch_range)
    
    # Plot
    fig, axes = plt.subplots(1, 2 if args.arousal_feature == "both" else 1, 
                              figsize=(14, 6) if args.arousal_feature == "both" else (8, 6))
    if args.arousal_feature != "both":
        axes = [axes]
    
    features = []
    if args.arousal_feature in ["rms_std", "both"]:
        features.append(("rms_std", "RMS Std (Energy Variation)"))
    if args.arousal_feature in ["pitch_range", "both"]:
        features.append(("pitch_range", "Pitch Range (Hz)"))
    
    for ax, (feat_key, feat_label) in zip(axes, features):
        for emotion in args.emotions:
            if emotion in data:
                x = np.array(data[emotion]["proj"])
                y = np.array(data[emotion][feat_key])
                ax.scatter(x, y, c=EMOTION_COLORS.get(emotion, "#999999"), 
                          label=f"{emotion} (n={len(x)})", alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add quadrant lines
        ax.axhline(y=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])/2, 
                   color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Labels
        ax.set_xlabel("Projection Score (Learned Axis)", fontsize=12, fontweight='bold')
        ax.set_ylabel(feat_label, fontsize=12, fontweight='bold')
        ax.set_title(f"Emotion Geometry: Learned Axis vs {feat_label}", fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.text(xlim[1]*0.7, ylim[1]*0.9, "High X\nHigh Arousal\n(Angry/Fearful)", 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(xlim[0]*0.7, ylim[0]*0.3, "Low X\nLow Arousal\n(Calm/Sad)", 
                ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] {args.out}")
    
    # Compute correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS (Learned Axis vs Acoustic Features)")
    print("-"*60)
    
    for emotion in args.emotions:
        if emotion in data and len(data[emotion]["proj"]) > 0:
            proj = np.array(data[emotion]["proj"])
            rms = np.array(data[emotion]["rms_std"])
            pitch = np.array(data[emotion]["pitch_range"])
            
            corr_rms = np.corrcoef(proj, rms)[0, 1]
            corr_pitch = np.corrcoef(proj, pitch)[0, 1]
            
            print(f"{emotion:12s}: RMS_std corr={corr_rms:+.3f}  |  Pitch_range corr={corr_pitch:+.3f}")
    
    # Overall correlation
    all_proj = []
    all_rms = []
    all_pitch = []
    for emotion in args.emotions:
        if emotion in data:
            all_proj.extend(data[emotion]["proj"])
            all_rms.extend(data[emotion]["rms_std"])
            all_pitch.extend(data[emotion]["pitch_range"])
    
    all_proj = np.array(all_proj)
    all_rms = np.array(all_rms)
    all_pitch = np.array(all_pitch)
    
    overall_corr_rms = np.corrcoef(all_proj, all_rms)[0, 1]
    overall_corr_pitch = np.corrcoef(all_proj, all_pitch)[0, 1]
    
    print("-"*60)
    print(f"{'OVERALL':12s}: RMS_std corr={overall_corr_rms:+.3f}  |  Pitch_range corr={overall_corr_pitch:+.3f}")
    print("="*60)
    
    print("\nINTERPRETATION:")
    if overall_corr_rms > 0.5 or overall_corr_pitch > 0.5:
        print("✓ STRONG correlation → Learned axis aligns with acoustic arousal!")
        print("  Your axis captures real prosodic energy, not just statistical artifact.")
    elif overall_corr_rms > 0.3 or overall_corr_pitch > 0.3:
        print("✓ MODERATE correlation → Axis partially captures arousal.")
        print("  May also encode other emotion dimensions (valence, etc.)")
    else:
        print("✗ WEAK correlation → Axis may capture non-prosodic features.")
        print("  Could be encoding semantic or speaker-specific patterns.")
    print("="*60)

if __name__ == "__main__":
    main()

