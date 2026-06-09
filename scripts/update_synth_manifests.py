#!/usr/bin/env python3
"""Add synthetic multilingual language entries to benchmark manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_LANGS = ["es", "nl", "fr", "it", "pl", "pt", "ru", "sv"]
MANIFESTS = [
    REPO_ROOT / "benchmark" / "models.json",
    REPO_ROOT / "benchmark" / "pro_models.json",
    REPO_ROOT / "benchmark" / "release_models.json",
]


def synth_language_entry(model_id: str, language: str) -> dict[str, Any]:
    return {
        "raw_file": f"predictions/raw/{model_id}/{language}.json",
        "source_file": "",
        "dataset": f"synth_verbatim_{language}",
        "splits": ["test"],
    }


def update_manifest(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    datasets = manifest.setdefault("datasets", {})
    for language in SYNTH_LANGS:
        datasets[language] = f"synth_verbatim_{language}"

    for model in manifest.get("models", []):
        model_id = model["id"]
        languages = model.setdefault("languages", {})
        for language in SYNTH_LANGS:
            languages[language] = synth_language_entry(model_id, language)

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated {path}")


def main() -> None:
    for manifest in MANIFESTS:
        update_manifest(manifest)


if __name__ == "__main__":
    main()
