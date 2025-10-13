#!/usr/bin/env python3
"""
Batch extract prosody features for all RAVDESS samples.
Processes happy (03) and angry (05) emotions for specified actors.
"""
import os
import glob
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch extract prosody from RAVDESS")
    parser.add_argument("--root", required=True, help="Root directory with Actor_XX folders")
    parser.add_argument("--actors", default="01-24", help="Actor range, e.g., '01-04' or '01-24'")
    parser.add_argument("--emotions", default="03,05", help="Emotion codes: 03=happy, 05=angry")
    parser.add_argument("--out_dir", default="embeddings", help="Output directory")
    args = parser.parse_args()
    
    # Parse actor range
    if "-" in args.actors:
        start, end = args.actors.split("-")
        actor_nums = range(int(start), int(end) + 1)
    else:
        actor_nums = [int(args.actors)]
    
    emotion_codes = args.emotions.split(",")
    emotion_names = {"03": "happy", "05": "angry", "01": "neutral", 
                     "02": "calm", "04": "sad", "06": "fearful", 
                     "07": "disgust", "08": "surprised"}
    
    total_files = 0
    processed_files = 0
    failed_files = []
    
    for actor_num in actor_nums:
        actor_id = f"Actor_{actor_num:02d}"
        actor_dir = os.path.join(args.root, actor_id)
        
        if not os.path.exists(actor_dir):
            print(f"[warn] {actor_dir} not found, skipping")
            continue
        
        # Find all WAV files
        wav_files = sorted(glob.glob(os.path.join(actor_dir, "*.wav")))
        
        for wav_path in wav_files:
            basename = os.path.basename(wav_path)
            parts = basename.split("-")
            
            if len(parts) < 7:
                continue
            
            emotion_code = parts[2]  # 03=happy, 05=angry, etc.
            intensity = parts[3]     # 01=normal, 02=strong
            
            # Filter by emotion
            if emotion_code not in emotion_codes:
                continue
            
            emotion_name = emotion_names.get(emotion_code, f"emotion{emotion_code}")
            
            # Create output path: embeddings/actor01/happy_01_sample01.npy
            out_subdir = os.path.join(args.out_dir, f"actor{actor_num:02d}")
            os.makedirs(out_subdir, exist_ok=True)
            
            # Create unique filename including intensity and sample number
            statement = parts[4]  # statement number
            repetition = parts[5]  # repetition number
            out_name = f"{emotion_name}_i{intensity}_s{statement}_r{repetition}_prosody.npy"
            out_path = os.path.join(out_subdir, out_name)
            
            # Skip if already exists
            if os.path.exists(out_path):
                print(f"[skip] {out_path} already exists")
                continue
            
            total_files += 1
            
            # Run extraction
            cmd = [
                "python3", "scripts/extract_prosody.py",
                "--wav", wav_path,
                "--sr", "16000",
                "--out", out_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    processed_files += 1
                    print(f"[{processed_files}/{total_files}] ✓ {out_path}")
                else:
                    failed_files.append((wav_path, result.stderr))
                    print(f"[{processed_files}/{total_files}] ✗ {wav_path}: {result.stderr}")
            except Exception as e:
                failed_files.append((wav_path, str(e)))
                print(f"[{processed_files}/{total_files}] ✗ {wav_path}: {e}")
    
    print("\n" + "="*70)
    print(f"SUMMARY: Processed {processed_files}/{total_files} files")
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for wav, error in failed_files[:10]:  # Show first 10 errors
            print(f"  {wav}: {error[:100]}")
    print("="*70)

if __name__ == "__main__":
    main()

