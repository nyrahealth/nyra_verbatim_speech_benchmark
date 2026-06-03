#!/usr/bin/env python3
"""Evaluate all cached normalized prediction JSONL files.

This script does not run model inference. It evaluates the JSONL files produced
by ``scripts/convert_outputs.py`` and writes one report directory per
model/language pair.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("outputs", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "predictions" / "normalized" / "manifest.json"),
        help="Manifest written by scripts/convert_outputs.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "results" / "cached"),
        help="Directory for evaluation reports.",
    )
    parser.add_argument("--language", choices=["en", "de", "all"], default="all")
    parser.add_argument("--model", default="", help="Optional model id to evaluate.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    rows = load_manifest(manifest_path)

    summary: List[Dict[str, Any]] = []
    for row in rows:
        model = row["model"]
        language = row["language"]
        if args.language != "all" and language != args.language:
            continue
        if args.model and model != args.model:
            continue

        pred_path = REPO_ROOT / row["normalized_file"]
        out_dir = output_root / model / language
        cmd = [
            sys.executable,
            str(REPO_ROOT / "evaluate.py"),
            "--predictions",
            str(pred_path),
            "--use_prediction_gold",
            "--output_dir",
            str(out_dir),
        ]
        print(f"Evaluating {model} / {language} ({row['num_records']} records)")
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

        report_path = out_dir / "report.json"
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)
        metrics = report.get("metrics", {})
        summary.append(
            {
                "model": model,
                "display_name": row.get("display_name", model),
                "language": language,
                "num_records": row["num_records"],
                "report": str(report_path.resolve().relative_to(REPO_ROOT)),
                "vWER": metrics.get("vWER"),
                "iWER": metrics.get("iWER"),
                "disfluency_f1": metrics.get("disfluency_f1"),
                "event_f1": metrics.get("event_f1"),
                "filler_f1": metrics.get("filler_f1"),
                "sound_f1": metrics.get("sound_f1"),
                "cutoff_f1": metrics.get("cutoff_f1"),
                "cutoff_lenient_f1": metrics.get("cutoff_lenient_f1"),
                "rep_f1": metrics.get("rep_f1"),
                "disfluency_omission_rate": metrics.get("disfluency_omission_rate"),
                "disfluency_hallucination_rate": metrics.get("disfluency_hallucination_rate"),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"results": summary}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
