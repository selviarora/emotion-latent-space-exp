#!/usr/bin/env python3
"""
Batch extract hubert-soft embeddings for all 24 actors.

Usage:
    python batch_extract_all.py --audio_root "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2"
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch extract HuBERT embeddings for all actors")
    parser.add_argument("--audio_root", required=True, help="Root directory containing Actor_01..Actor_24")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    args = parser.parse_args()
    
    audio_root = Path(args.audio_root)
    project_root = Path(__file__).parent.parent.parent  # emotion_expirement/
    embeddings_dir = project_root / "embeddings"
    extractor_script = Path(__file__).parent / "extract_hubert_soft.py"
    
    # Process actors 01-24
    for i in range(1, 25):
        actor_id = f"{i:02d}"
        input_dir = audio_root / f"Actor_{actor_id}"
        output_dir = embeddings_dir / f"actor{actor_id}"
        
        if not input_dir.exists():
            print(f"[SKIP] Actor {actor_id}: {input_dir} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Actor {actor_id} ({i}/24)")
        print(f"{'='*60}")
        
        # Call extract_hubert_soft.py
        cmd = [
            sys.executable,
            str(extractor_script),
            "--input_dir", str(input_dir),
            "--output_dir", str(output_dir),
            "--device", args.device
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to process Actor {actor_id}")
            continue
    
    print(f"\n{'='*60}")
    print("✅ Batch extraction complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

