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
    ("Disfluency F1 ↑", "disfluency_f1"),
    ("Filler F1↑", "filler_f1"),
    ("Vocal Sound F1↑", "sound_f1"),
    ("Cutoff F1↑", "cutoff_lenient_f1"),
    ("Rep F1↑", "rep_f1"),
    ("vWER↓", "vWER"),
]

PROVIDER_BADGE_CONFIG = {
    "assemblyai_p1": ("AssemblyAI", "111111", "assemblyai", "white"),
    "assemblyai_p2": ("AssemblyAI", "111111", "assemblyai", "white"),
    "canary": ("NVIDIA", "76B900", "nvidia", "white"),
    "cartesia": ("Cartesia", "5B5BD6", "", "white"),
    "cohere": ("Cohere", "39594D", "cohere", "white"),
    "crisperwhisper_1": ("Nyra Health", "0F766E", "", "white"),
    "crisperwhisper_2": ("Nyra Health", "0F766E", "", "white"),
    "deepgram": ("Deepgram", "13EF93", "deepgram", "black"),
    "elevenlabs": ("ElevenLabs", "000000", "elevenlabs", "white"),
    "fish_audio": ("Fish Audio", "2563EB", "", "white"),
    "gradium": ("Gradium", "7C3AED", "", "white"),
    "inworld": ("Inworld", "FF6B00", "", "white"),
    "mai_transcribe_15": ("Microsoft", "5E5E5E", "microsoft", "white"),
    "whisper_large_v3": ("OpenAI", "412991", "openai", "white"),
    "xai_grok": ("xAI", "111111", "", "white"),
}


def provider_badge(model_id: str) -> str:
    config = PROVIDER_BADGE_CONFIG.get(model_id)
    if not config:
        return ""
    label, color, logo, logo_color = config
    escaped_label = label.replace(" ", "%20")
    query = "style=flat-square"
    if logo:
        query = f"{query}&logo={logo}&logoColor={logo_color}"
    return (
        f"![{label}]"
        f"(https://img.shields.io/badge/-{escaped_label}-{color}?{query})"
    )


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


def model_label(row: Dict[str, Any], names: Dict[str, str]) -> str:
    model_id = row["model"]
    name = names.get(model_id, model_id)
    badge = provider_badge(model_id)
    return f"{badge} {name}" if badge else name


def print_table(
    rows: List[Dict[str, Any]],
    language: str,
    title: str,
    names: Dict[str, str],
) -> str:
    selected = [row for row in rows if row["language"] == language]
    selected.sort(
        key=lambda row: (
            row.get("disfluency_f1")
            if row.get("disfluency_f1") is not None else -1
        ),
        reverse=True,
    )

    lines = [f"## {title}", ""]
    header = "| Model | n | " + " | ".join(label for label, _ in METRICS) + " |"
    align = "|---|---:|" + "|".join("---:" for _ in METRICS) + "|"
    lines.extend([header, align])
    for row in selected:
        values = [fmt(row.get(key)) for _, key in METRICS]
        model_name = model_label(row, names)
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
