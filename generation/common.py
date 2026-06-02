"""Shared helpers for raw prediction generation examples."""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import soundfile as sf
from datasets import Audio, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = REPO_ROOT / "benchmark" / "models.json"

DATASETS = {
    "en": "nyrahealth/disfluency_speech_english",
    "de": "nyrahealth/disfluency_speech_german",
}
DEFAULT_SPLITS = {
    "en": ["train", "validation", "test"],
    "de": ["test"],
}


TranscriptFn = Callable[[Path, str], Optional[str]]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--language", choices=["en", "de", "all"], default="all")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to process. Defaults to all benchmarked splits.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--output_path",
        default="",
        help="Optional explicit output JSON path. By default, uses benchmark/models.json.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start from an empty output file instead of appending missing samples.",
    )


def languages_from_arg(language: str) -> List[str]:
    return ["en", "de"] if language == "all" else [language]


def manifest_raw_path(model_id: str, language: str) -> Path:
    config = manifest_language_config(model_id, language)
    return REPO_ROOT / config["raw_file"]


def manifest_language_config(model_id: str, language: str) -> Dict[str, Any]:
    with MODELS_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)
    for model in manifest.get("models", []):
        if model.get("id") == model_id:
            config = model.get("languages", {}).get(language)
            if config:
                return config
    raise ValueError(f"No language config found for {model_id}/{language}")


def load_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def iter_dataset_rows(
    language: str,
    splits: Optional[Iterable[str]],
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    split_list = list(splits) if splits else DEFAULT_SPLITS[language]
    emitted = 0
    for split in split_list:
        dataset = load_dataset(DATASETS[language], split=split)
        dataset = dataset.cast_column("audio", Audio(decode=True))
        for row in dataset:
            row = dict(row)
            row["split"] = row.get("split") or split
            yield row
            emitted += 1
            if max_samples and emitted >= max_samples:
                return


def sample_id(row: Dict[str, Any]) -> str:
    audio = row.get("audio")
    if row.get("id"):
        return str(row["id"])
    if isinstance(audio, dict) and audio.get("path"):
        return Path(audio["path"]).stem
    return str(row.get("audio", "sample"))


def audio_name(row: Dict[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    if isinstance(audio, str):
        return audio
    return f"{sample_id(row)}.wav"


@contextlib.contextmanager
def wav_path(row: Dict[str, Any]) -> Iterator[Path]:
    """Return a WAV path for a Hugging Face audio row."""
    audio = row["audio"]
    if isinstance(audio, dict) and audio.get("path") and Path(audio["path"]).exists():
        yield Path(audio["path"])
        return
    if isinstance(audio, str) and Path(audio).exists():
        yield Path(audio)
        return

    if not isinstance(audio, dict) or "array" not in audio:
        raise ValueError("Expected decoded Hugging Face Audio field")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio["array"], audio["sampling_rate"])
        yield Path(tmp.name)


def raw_record(
    row: Dict[str, Any],
    *,
    dataset_name: str,
    language: str,
    model_id: str,
    prediction: Optional[str],
    intended_prediction: Optional[str] = None,
) -> Dict[str, Any]:
    gold_intended = row.get("intended_transcript") or row.get("transcript") or ""
    return {
        "id": sample_id(row),
        "audio": audio_name(row),
        "dataset": dataset_name,
        "split": str(row.get("split", "")),
        "language": language,
        "model": model_id,
        "raw_verbatim_prediction": prediction,
        "raw_intended_prediction": intended_prediction,
        "gold_verbatim": row.get("verbatim_transcript", "") or "",
        "gold_intended": gold_intended,
    }


def generate_raw_predictions(
    *,
    model_id: str,
    language: str,
    splits: Optional[Iterable[str]],
    max_samples: int,
    output_path: Optional[Path],
    resume: bool,
    transcribe: TranscriptFn,
) -> Path:
    manifest_config = manifest_language_config(model_id, language)
    effective_splits = list(splits) if splits else manifest_config.get("splits")
    out_path = output_path or manifest_raw_path(model_id, language)
    rows = load_existing(out_path) if resume else []
    done = {str(row.get("id")) for row in rows if row.get("raw_verbatim_prediction")}

    for index, sample in enumerate(
        iter_dataset_rows(language, splits=effective_splits, max_samples=max_samples),
        start=1,
    ):
        sid = sample_id(sample)
        if sid in done:
            continue
        print(f"[{language}] {index}: {sid}")
        with wav_path(sample) as audio_path:
            prediction = transcribe(audio_path, language)
        rows.append(
            raw_record(
                sample,
                dataset_name=DATASETS[language],
                language=language,
                model_id=model_id,
                prediction=prediction,
            )
        )
        write_json(out_path, rows)

    write_json(out_path, rows)
    return out_path
