#!/usr/bin/env python3
"""Convert cached model outputs into benchmark prediction JSONL files.

The converter is manifest-driven. Model metadata lives in
``benchmark/models.json`` so adding a model should not require editing this
script. The script can import legacy experiment outputs into the fixed raw
prediction schema and then normalize those raw predictions into evaluator-ready
JSONL files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from normalize import normalize_prediction  # noqa: E402  pylint: disable=wrong-import-position,import-error

DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "models.json"


def infer_id(audio_path: str) -> str:
    return Path(audio_path).stem


def infer_split(sample_id: str, language: str) -> str:
    upper = sample_id.upper()
    if language == "en":
        for split in ("TRAIN", "VALIDATION", "TEST"):
            marker = f"DISFLUENCY_{split}_"
            if marker in upper:
                return split.lower()
    # The current German cached outputs are the test set.
    return "test"


def _first_nonempty(obj: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def _identity(text: str, language: str) -> str:
    del language
    return text or ""


def _assemblyai_source_adapter(text: str, language: str) -> str:
    """Handle AssemblyAI-only source quirks before global normalization."""
    import re

    del language
    out = text or ""
    # Laughter forms observed in AssemblyAI output. These are not global
    # because bare "ah" can be a normal interjection in other models.
    out = re.sub(r"\ba?haha+\b", "[laughter]", out, flags=re.IGNORECASE)
    out = re.sub(
        r"\b(?:hah|hehe|hihi)\b",
        "[laughter]",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\bah\b", "[laughter]", out, flags=re.IGNORECASE)
    return out


SOURCE_ADAPTERS: Dict[str, Callable[[str, str], str]] = {
    "assemblyai": _assemblyai_source_adapter,
}


def normalize_for_model(text: str, adapter_name: str, language: str) -> str:
    adapter = SOURCE_ADAPTERS.get(adapter_name, _identity)
    return normalize_prediction(adapter(text or "", language), language)


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_model_languages(
    manifest: Dict[str, Any],
) -> Iterable[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
    for model in manifest.get("models", []):
        for language, config in (model.get("languages") or {}).items():
            yield model, language, config


def legacy_to_raw_record(
    raw: Dict[str, Any],
    model: Dict[str, Any],
    language: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    audio = _first_nonempty(raw, ["audio", "audio_path", "path"])
    sample_id = str(raw.get("id") or infer_id(audio))
    split = str(raw.get("split") or infer_split(sample_id, language))

    raw_verbatim = _first_nonempty(
        raw,
        [
            "raw_verbatim_prediction",
            "pred_verbatim",
            "verbatim_prediction",
            "prediction",
            "transcript",
            "text",
        ],
    )
    raw_intended = _first_nonempty(
        raw,
        [
            "raw_intended_prediction",
            "pred_intended",
            "intended_prediction",
        ],
    )
    if not raw_intended and model.get("single_output", True):
        raw_intended = raw_verbatim

    return {
        "id": sample_id,
        "audio": audio,
        "dataset": config.get("dataset", ""),
        "split": split,
        "language": language,
        "model": model["id"],
        "source_file": config.get("source_file", ""),
        "raw_verbatim_prediction": raw_verbatim,
        "raw_intended_prediction": raw_intended,
        "gold_verbatim": _first_nonempty(
            raw,
            ["gold_verbatim", "verbatim_ground_truth", "verbatim_transcript"],
        ),
        "gold_intended": _first_nonempty(
            raw,
            [
                "gold_intended",
                "intended_ground_truth",
                "intended_transcript",
                "transcript",
            ],
        ),
    }


def normalize_raw_record(
    raw: Dict[str, Any],
    model: Dict[str, Any],
) -> Dict[str, Any]:
    adapter_name = str(model.get("adapter", "default"))
    language = str(raw["language"])
    raw_verbatim = raw.get("raw_verbatim_prediction") or ""
    raw_intended = raw.get("raw_intended_prediction")
    if raw_intended is None:
        raw_intended = raw_verbatim

    out = dict(raw)
    out["pred_verbatim"] = normalize_for_model(
        raw_verbatim,
        adapter_name,
        language,
    )
    out["pred_intended"] = normalize_for_model(
        str(raw_intended),
        adapter_name,
        language,
    )
    return out


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, records: List[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        default="/home/ubuntu/code/research/evaluation",
        help=(
            "Optional directory containing legacy source files from "
            "models.json."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to benchmark/models.json.",
    )
    parser.add_argument(
        "--skip_import",
        action="store_true",
        help=(
            "Do not import legacy source files; read fixed raw files "
            "directly."
        ),
    )
    parser.add_argument(
        "--normalized_dir",
        default=str(REPO_ROOT / "predictions" / "normalized"),
        help="Destination for canonical JSONL files.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model id to convert.",
    )
    parser.add_argument(
        "--language",
        choices=["en", "de", "all"],
        default="all",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    normalized_dir = Path(args.normalized_dir)
    model_manifest = load_manifest(Path(args.manifest))

    converted_manifest: List[Dict[str, Any]] = []
    for model, language, config in iter_model_languages(model_manifest):
        model_id = model["id"]
        if args.model and model_id != args.model:
            continue
        if args.language != "all" and language != args.language:
            continue

        raw_path = REPO_ROOT / config["raw_file"]
        if not args.skip_import:
            source_name = config.get("source_file")
            if source_name:
                source_path = source_dir / source_name
                if not source_path.exists():
                    raise FileNotFoundError(source_path)
                with source_path.open(encoding="utf-8") as f:
                    legacy_records = json.load(f)
                if not isinstance(legacy_records, list):
                    raise ValueError(f"Expected list in {source_path}")
                raw_records = [
                    legacy_to_raw_record(record, model, language, config)
                    for record in legacy_records
                ]
                if config.get("drop_empty_predictions"):
                    raw_records = [
                        record
                        for record in raw_records
                        if (record.get("raw_verbatim_prediction") or "").strip()
                    ]
                write_json(raw_path, raw_records)

        with raw_path.open(encoding="utf-8") as f:
            raw_records = json.load(f)
        if not isinstance(raw_records, list):
            raise ValueError(f"Expected list in {raw_path}")

        converted = [normalize_raw_record(record, model) for record in raw_records]
        out_path = normalized_dir / model_id / f"{language}.jsonl"
        count = write_jsonl(out_path, converted)

        converted_manifest.append(
            {
                "model": model_id,
                "display_name": model.get("display_name", model_id),
                "language": language,
                "dataset": config.get("dataset"),
                "raw_file": str(raw_path.relative_to(REPO_ROOT)),
                "source_file": config.get("source_file", ""),
                "normalized_file": str(
                    out_path.relative_to(REPO_ROOT),
                ),
                "num_records": count,
                "splits": sorted({r["split"] for r in converted}),
            }
        )
        print(
            f"{model_id:20s} {language} {count:5d} -> {out_path}",
        )

    manifest_path = normalized_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"outputs": converted_manifest}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
