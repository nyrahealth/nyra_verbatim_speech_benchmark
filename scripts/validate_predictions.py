#!/usr/bin/env python3
"""Validate model manifests and cached prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from normalize import normalize_prediction


DEFAULT_MODEL_MANIFEST = REPO_ROOT / "benchmark" / "models.json"
DEFAULT_NORMALIZED_MANIFEST = REPO_ROOT / "predictions" / "normalized" / "manifest.json"


class ValidationError(Exception):
    """Raised when validation fails."""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


RAW_REQUIRED = {
    "id",
    "audio",
    "dataset",
    "split",
    "language",
    "model",
    "raw_verbatim_prediction",
    "gold_verbatim",
    "gold_intended",
}

NORMALIZED_REQUIRED = RAW_REQUIRED | {"pred_verbatim", "pred_intended"}


def check_required(row: Dict[str, Any], required: Iterable[str], path: Path, idx: int) -> None:
    missing = [key for key in required if key not in row]
    if missing:
        raise ValidationError(f"{path}: record {idx} missing required fields: {missing}")
    for key in ("id", "dataset", "split", "language", "model", "gold_verbatim", "gold_intended"):
        if not str(row.get(key, "")).strip():
            raise ValidationError(f"{path}: record {idx} has empty field {key!r}")


def validate_unique_ids(rows: List[Dict[str, Any]], path: Path) -> None:
    seen = set()
    duplicates = set()
    for row in rows:
        sid = str(row.get("id", ""))
        if sid in seen:
            duplicates.add(sid)
        seen.add(sid)
    if duplicates:
        sample = sorted(duplicates)[:5]
        raise ValidationError(f"{path}: duplicate ids found: {sample}")


def validate_raw_file(path: Path, model_id: str, language: str) -> int:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValidationError(f"{path}: expected a JSON list")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValidationError(f"{path}: record {idx} is not an object")
        check_required(row, RAW_REQUIRED, path, idx)
        if row["model"] != model_id or row["language"] != language:
            raise ValidationError(
                f"{path}: record {idx} model/language mismatch "
                f"({row['model']}/{row['language']})"
            )
    validate_unique_ids(rows, path)
    return len(rows)


def validate_normalized_file(path: Path, model_id: str, language: str) -> int:
    rows = load_jsonl(path)
    for idx, row in enumerate(rows):
        check_required(row, NORMALIZED_REQUIRED, path, idx)
        if row["model"] != model_id or row["language"] != language:
            raise ValidationError(
                f"{path}: record {idx} model/language mismatch "
                f"({row['model']}/{row['language']})"
            )
        normalized = normalize_prediction(row["pred_verbatim"], language)
        if normalized != row["pred_verbatim"]:
            raise ValidationError(
                f"{path}: record {idx} pred_verbatim is not idempotent under "
                "normalize_prediction"
            )
    validate_unique_ids(rows, path)
    return len(rows)


def validate_model_manifest(path: Path) -> Dict[str, Any]:
    manifest = load_json(path)
    ids = set()
    for model in manifest.get("models", []):
        model_id = model.get("id")
        if not model_id:
            raise ValidationError(f"{path}: model without id")
        if model_id in ids:
            raise ValidationError(f"{path}: duplicate model id {model_id}")
        ids.add(model_id)
        if model.get("adapter", "default") not in {"default", "assemblyai"}:
            raise ValidationError(f"{path}: unknown adapter for {model_id}")
        if not model.get("languages"):
            raise ValidationError(f"{path}: model {model_id} has no languages")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--normalized_manifest", default=str(DEFAULT_NORMALIZED_MANIFEST))
    parser.add_argument("--skip_normalized", action="store_true")
    args = parser.parse_args()

    model_manifest = validate_model_manifest(Path(args.models))
    print(f"Validated model manifest: {args.models}")

    for model in model_manifest.get("models", []):
        model_id = model["id"]
        for language, config in (model.get("languages") or {}).items():
            raw_path = REPO_ROOT / config["raw_file"]
            count = validate_raw_file(raw_path, model_id, language)
            print(f"raw {model_id}/{language}: {count}")

    if not args.skip_normalized:
        normalized_manifest = load_json(Path(args.normalized_manifest))
        for row in normalized_manifest.get("outputs", []):
            path = REPO_ROOT / row["normalized_file"]
            count = validate_normalized_file(path, row["model"], row["language"])
            if count != row.get("num_records"):
                raise ValidationError(
                    f"{path}: manifest count {row.get('num_records')} != {count}"
                )
            print(f"normalized {row['model']}/{row['language']}: {count}")

    print("Validation passed.")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
