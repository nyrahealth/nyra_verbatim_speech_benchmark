#!/usr/bin/env python3
"""
Convenience one-shot runner: transcribe + evaluate in sequence.

Usage::

    python run.py \\
        --model /path/to/CrisperWhisper \\
        --datasets nyrahealth/disfluency_speech_english \\
        --splits test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run transcription and evaluation end-to-end."
    )
    # Transcription args
    parser.add_argument("--model", required=True, help="Path or HF id of the model.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--modes", nargs="+", default=["verbatim", "intended"])
    parser.add_argument("--tag_count", type=int, default=5)
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="")
    # Evaluation args
    parser.add_argument("--cutoff_source", default="*")
    parser.add_argument("--cutoff_target", default="-")
    parser.add_argument("--verbatim_key", default="verbatim_transcript")
    parser.add_argument("--intended_key", default="intended_transcript")
    # Output
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Transcribe
    print("=" * 60)
    print("Step 1: Transcription")
    print("=" * 60)
    transcribe_cmd = [
        sys.executable, os.path.join(repo_dir, "transcribe.py"),
        "--model", args.model,
        "--datasets", *args.datasets,
        "--splits", *args.splits,
        "--modes", *args.modes,
        "--tag_count", str(args.tag_count),
        "--language", args.language,
        "--batch_size", str(args.batch_size),
        "--num_beams", str(args.num_beams),
        "--max_new_tokens", str(args.max_new_tokens),
        "--output", pred_path,
    ]
    if args.device:
        transcribe_cmd.extend(["--device", args.device])

    ret = subprocess.run(transcribe_cmd)
    if ret.returncode != 0:
        print(f"Transcription failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

    # Step 2: Evaluate
    print()
    print("=" * 60)
    print("Step 2: Evaluation")
    print("=" * 60)
    evaluate_cmd = [
        sys.executable, os.path.join(repo_dir, "evaluate.py"),
        "--predictions", pred_path,
        "--datasets", *args.datasets,
        "--splits", *args.splits,
        "--cutoff_source", args.cutoff_source,
        "--cutoff_target", args.cutoff_target,
        "--verbatim_key", args.verbatim_key,
        "--intended_key", args.intended_key,
        "--output_dir", args.output_dir,
    ]
    ret = subprocess.run(evaluate_cmd)
    if ret.returncode != 0:
        print(f"Evaluation failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

    print()
    print("All done. Results in:", args.output_dir)


if __name__ == "__main__":
    main()
