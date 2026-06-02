#!/usr/bin/env python3
"""Print markdown leaderboards from cached evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = REPO_ROOT / "benchmark" / "models.json"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "cached" / "summary.json"


METRICS = [
    ("vWER↓", "vWER"),
    ("iWER↓", "iWER"),
    ("Filler F1↑", "filler_f1"),
    ("Sound F1↑", "sound_f1"),
    ("Cutoff F1↑", "cutoff_f1"),
    ("Cutoff Lenient F1↑", "cutoff_lenient_f1"),
    ("Rep F1↑", "rep_f1"),
    ("Omit↓", "disfluency_omission_rate"),
    ("Halluc↓", "disfluency_hallucination_rate"),
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def display_names(models_path: Path) -> Dict[str, str]:
    data = load_json(models_path)
    return {
        model["id"]: model.get("display_name", model["id"])
        for model in data.get("models", [])
    }


def print_table(
    rows: List[Dict[str, Any]],
    language: str,
    title: str,
    names: Dict[str, str],
) -> str:
    selected = [row for row in rows if row["language"] == language]
    selected.sort(key=lambda row: row.get("vWER") if row.get("vWER") is not None else 999)

    lines = [f"## {title}", ""]
    header = "| Model | n | " + " | ".join(label for label, _ in METRICS) + " |"
    align = "|---|---:|" + "|".join("---:" for _ in METRICS) + "|"
    lines.extend([header, align])
    for row in selected:
        values = [fmt(row.get(key)) for _, key in METRICS]
        model_name = names.get(row["model"], row["model"])
        lines.append(
            f"| {model_name} | {row['num_records']} | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--models", default=str(DEFAULT_MODELS))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = load_json(Path(args.summary)).get("results", [])
    names = display_names(Path(args.models))
    text = "\n\n".join(
        [
            print_table(rows, "en", "English", names),
            print_table(rows, "de", "German", names),
        ]
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
