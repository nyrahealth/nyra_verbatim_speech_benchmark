"""Shared helpers for raw prediction generation examples."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import soundfile as sf
from datasets import Audio, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = REPO_ROOT / "benchmark" / "models.json"
SECRETS_PATH = REPO_ROOT / "secrets.env"


def load_secrets(path: Path = SECRETS_PATH) -> None:
    """Export ``KEY=VALUE`` lines from ``secrets.env`` into ``os.environ``.

    Copy ``secrets.env.example`` to ``secrets.env`` (gitignored) and fill in
    the API keys for the hosted-model scripts you want to run.  Variables
    already present in the environment take precedence over the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


# Loaded at import time so every generation script's os.environ.get() default
# sees the keys without per-script wiring.
load_secrets()

DATASETS = {
    "en": "nyralabs/disfluency_speech_english",
    "de": "nyralabs/disfluency_speech_german",
}
DEFAULT_SPLITS = {
    "en": ["train", "validation", "test"],
    "de": ["test"],
}

SYNTH_LANGUAGES = ["es", "nl", "fr", "it", "pl", "pt", "ru", "sv"]
SYNTH_DATASETS = {
    "es": {
        "name": "synth_verbatim_es",
        "root": Path("/home/ubuntu/data/spanish_speech/synth_verbatim_es"),
    },
    "nl": {
        "name": "synth_verbatim_nl",
        "root": Path("/home/ubuntu/data/dutch_speech/synth_verbatim_nl"),
    },
    "fr": {
        "name": "synth_verbatim_fr",
        "root": Path("/home/ubuntu/data/french_speech/synth_verbatim_fr"),
    },
    "it": {
        "name": "synth_verbatim_it",
        "root": Path("/home/ubuntu/data/italian_speech/synth_verbatim_it"),
    },
    "pl": {
        "name": "synth_verbatim_pl",
        "root": Path("/home/ubuntu/data/polish_speech/synth_verbatim_pl"),
    },
    "pt": {
        "name": "synth_verbatim_pt",
        "root": Path("/home/ubuntu/data/portuguese_speech/synth_verbatim_pt"),
    },
    "ru": {
        "name": "synth_verbatim_ru",
        "root": Path("/home/ubuntu/data/russian_speech/synth_verbatim_ru"),
    },
    "sv": {
        "name": "synth_verbatim_sv",
        "root": Path("/home/ubuntu/data/swedish_speech/synth_verbatim_sv"),
    },
}
DEFAULT_SYNTH_SPLITS = ["test"]

LANGUAGE_CHOICES = [
    "en",
    "de",
    *SYNTH_LANGUAGES,
    "all",
    "synth_all",
]


TranscriptFn = Callable[[Path, str], Optional[str]]


def is_synth_language(language: str) -> bool:
    return language in SYNTH_DATASETS


def dataset_name_for_language(language: str) -> str:
    if is_synth_language(language):
        return SYNTH_DATASETS[language]["name"]
    return DATASETS[language]


def default_raw_output_path(model_id: str, language: str) -> Path:
    return REPO_ROOT / "predictions" / "raw" / model_id / f"{language}.json"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--language", choices=LANGUAGE_CHOICES, default="all")
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
    if language == "all":
        return ["en", "de"]
    if language == "synth_all":
        return list(SYNTH_LANGUAGES)
    return [language]


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


def _iter_synth_rows(
    language: str,
    splits: Iterable[str],
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    config = SYNTH_DATASETS[language]
    labels_path = config["root"] / "labels.json"
    with labels_path.open(encoding="utf-8") as f:
        all_rows = json.load(f)

    split_set = set(splits)
    emitted = 0
    for row in all_rows:
        row_split = str(row.get("split", ""))
        if row_split not in split_set:
            continue
        sample = dict(row)
        sample["split"] = row_split
        sample["_dataset_root"] = str(config["root"])
        yield sample
        emitted += 1
        if max_samples and emitted >= max_samples:
            return


def iter_dataset_rows(
    language: str,
    splits: Optional[Iterable[str]],
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    if is_synth_language(language):
        split_list = list(splits) if splits else DEFAULT_SYNTH_SPLITS
        yield from _iter_synth_rows(language, split_list, max_samples)
        return

    split_list = list(splits) if splits else DEFAULT_SPLITS[language]
    emitted = 0
    for split in split_list:
        dataset = load_dataset(DATASETS[language], split=split)
        # decode=False keeps the raw (wav) bytes so we can decode with
        # soundfile and avoid the optional torchcodec dependency.
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for row in dataset:
            row = dict(row)
            row["split"] = row.get("split") or split
            yield row
            emitted += 1
            if max_samples and emitted >= max_samples:
                return


def sample_id(row: Dict[str, Any]) -> str:
    audio = row.get("audio")
    if row.get("id") is not None:
        return str(row["id"])
    if isinstance(audio, dict) and audio.get("path"):
        return Path(audio["path"]).stem
    if isinstance(audio, str):
        return Path(audio).stem
    return str(row.get("audio", "sample"))


def audio_name(row: Dict[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    if isinstance(audio, str):
        return audio
    return f"{sample_id(row)}.wav"


def resolve_audio_path(row: Dict[str, Any]) -> Optional[Path]:
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        path = Path(audio["path"])
        if path.exists():
            return path
    if isinstance(audio, str):
        rel = Path(audio)
        root = row.get("_dataset_root")
        if root:
            candidate = Path(root) / rel
            if candidate.exists():
                return candidate
        if rel.exists():
            return rel
    return None


@contextlib.contextmanager
def wav_path(row: Dict[str, Any]) -> Iterator[Path]:
    """Return a WAV path for a Hugging Face or local synth audio row."""
    resolved = resolve_audio_path(row)
    if resolved is not None:
        yield resolved
        return

    audio = row["audio"]
    if isinstance(audio, dict) and audio.get("path") and Path(audio["path"]).exists():
        yield Path(audio["path"])
        return
    if isinstance(audio, str) and Path(audio).exists():
        yield Path(audio)
        return

    # Undecoded Hugging Face Audio field: raw (wav) bytes from decode=False.
    if isinstance(audio, dict) and audio.get("bytes"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio["bytes"])
            tmp.flush()
            yield Path(tmp.name)
        return

    if not isinstance(audio, dict) or "array" not in audio:
        raise ValueError("Expected decoded Hugging Face Audio field or local audio path")

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


def resolve_generation_config(
    *,
    model_id: str,
    language: str,
    splits: Optional[Iterable[str]],
    output_path: Optional[Path],
) -> tuple[Path, List[str], str]:
    if is_synth_language(language):
        effective_splits = list(splits) if splits else DEFAULT_SYNTH_SPLITS
        out_path = output_path or default_raw_output_path(model_id, language)
        dataset_name = dataset_name_for_language(language)
        return out_path, effective_splits, dataset_name

    # When the caller provides both splits and an explicit output path there is
    # no need to consult benchmark/models.json (lets models that live in other
    # manifests, e.g. pro_models.json, generate en/de directly).
    if splits and output_path is not None:
        return output_path, list(splits), DATASETS[language]

    manifest_config = manifest_language_config(model_id, language)
    effective_splits = list(splits) if splits else manifest_config.get("splits")
    out_path = output_path or manifest_raw_path(model_id, language)
    dataset_name = DATASETS[language]
    return out_path, effective_splits, dataset_name


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
    out_path, effective_splits, dataset_name = resolve_generation_config(
        model_id=model_id,
        language=language,
        splits=splits,
        output_path=output_path,
    )
    rows = load_existing(out_path) if resume else []
    # Drop cached rows with empty predictions so resume can refetch them.
    rows = [row for row in rows if row.get("raw_verbatim_prediction")]
    done = {str(row.get("id")) for row in rows}

    for index, sample in enumerate(
        iter_dataset_rows(language, splits=effective_splits, max_samples=max_samples),
        start=1,
    ):
        sid = sample_id(sample)
        if sid in done:
            continue
        print(f"[{language}] {index}: {sid}", flush=True)
        with wav_path(sample) as audio_path:
            prediction = transcribe(audio_path, language)
        rows.append(
            raw_record(
                sample,
                dataset_name=dataset_name,
                language=language,
                model_id=model_id,
                prediction=prediction,
            )
        )
        write_json(out_path, rows)

    write_json(out_path, rows)
    return out_path
