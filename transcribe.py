#!/usr/bin/env python3
"""
Verbatim ASR Transcription — Step 1.

Loads audio from a HuggingFace dataset and runs a Whisper-family model in
verbatim and/or intended mode, caching results to a JSONL file.

Supports both HuggingFace Hub datasets and local ``load_from_disk`` paths.

Usage::

    python transcribe.py \\
        --model /path/to/CrisperWhisper \\
        --datasets nyrahealth/disfluency_speech_english \\
        --splits test \\
        --modes verbatim intended \\
        --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np


def _resolve_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def _load_model(model_path: str, device: str):
    """Return (model, processor, torch_device, torch_dtype)."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch_device = torch.device(device)
    dtype = torch.float16 if "cuda" in device else torch.float32

    processor = None
    candidates = [model_path]
    base = os.path.basename(os.path.normpath(model_path))
    parent = os.path.dirname(os.path.normpath(model_path))
    if base.startswith("checkpoint-") and parent and os.path.isdir(parent):
        candidates.append(parent)
    for cand in candidates:
        try:
            processor = WhisperProcessor.from_pretrained(cand)
            break
        except Exception:
            continue
    if processor is None:
        raise RuntimeError(f"Could not load processor from: {candidates}")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(torch_device)
    model.eval()

    return model, processor, torch_device, dtype


def _build_prompt(mode: str, tag_count: int) -> str:
    if mode == "verbatim":
        return "".join(f"[verbatim_{i}]" for i in range(1, tag_count + 1))
    elif mode == "intended":
        return "".join(f"[intended_{i}]" for i in range(1, tag_count + 1))
    return ""


def _decoder_prefix_with_language(processor, model, language: Optional[str]) -> List[int]:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return []
    dec_start = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
    if dec_start is None:
        dec_start = getattr(tok, "bos_token_id", None)
    prefix = [int(dec_start)] if dec_start is not None else []
    language = (language or "").strip()
    if not language:
        return prefix
    if hasattr(processor, "get_decoder_prompt_ids"):
        forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        prefix += [int(tid) for _, tid in forced]
    return prefix


def _transcribe_batch(
    model, processor, device, dtype,
    audio_arrays: List[np.ndarray],
    sampling_rate: int,
    prompt_ids: List[int],
    prefix_ids: List[int],
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    import torch

    inputs = processor.feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    decoder_input_ids = None
    row = prompt_ids + prefix_ids
    if row:
        decoder_input_ids = torch.tensor(
            [row] * len(audio_arrays), dtype=torch.long, device=device,
        )

    with torch.inference_mode():
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            forced_decoder_ids=None,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [(t or "").strip() for t in texts]


def _extract_audio_array(audio_field: Any, target_sr: int) -> np.ndarray:
    """
    Extract a numpy audio array from a HuggingFace Audio field.

    Handles three cases:
      1. Decoded dict with ``{"array": ..., "sampling_rate": ...}``
      2. Raw dict with ``{"bytes": ..., "path": ...}`` (decode=False or broken decoder)
      3. File path string
    """
    import soundfile as sf
    import librosa

    if isinstance(audio_field, dict):
        if "array" in audio_field and audio_field["array"] is not None:
            arr = np.asarray(audio_field["array"], dtype=np.float32)
            file_sr = audio_field.get("sampling_rate", target_sr)
            if file_sr != target_sr:
                arr = librosa.resample(arr, orig_sr=file_sr, target_sr=target_sr)
            return arr

        if "bytes" in audio_field and audio_field["bytes"] is not None:
            arr, file_sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if file_sr != target_sr:
                arr = librosa.resample(arr, orig_sr=file_sr, target_sr=target_sr)
            return arr

        if "path" in audio_field and audio_field["path"]:
            arr, _ = librosa.load(audio_field["path"], sr=target_sr, mono=True)
            return arr

    if isinstance(audio_field, str):
        arr, _ = librosa.load(audio_field, sr=target_sr, mono=True)
        return arr

    raise ValueError(f"Cannot extract audio from: {type(audio_field)}")


def _load_hf_dataset(ds_name: str, split: str):
    """Load a dataset split, supporting both Hub names and local paths."""
    if os.path.isdir(ds_name):
        from datasets import load_from_disk
        ds_dict = load_from_disk(ds_name)
        return ds_dict[split]
    else:
        from datasets import load_dataset
        return load_dataset(ds_name, split=split)


def _get_audio_for_row(ds, idx: int, use_raw: bool):
    """
    Get audio data for a dataset row.  When ``use_raw`` is True, bypasses the
    Audio decoder and returns the raw ``{bytes, path}`` dict instead.
    """
    if use_raw:
        from datasets import Audio
        ds_raw = ds.cast_column("audio", Audio(decode=False))
        return ds_raw[idx]["audio"]
    return ds[idx]["audio"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with a Whisper-family model.")
    parser.add_argument("--model", required=True, help="Path or HF id of the model.")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="HuggingFace dataset name(s) or local load_from_disk paths.")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--modes", nargs="+", default=["verbatim", "intended"],
                        choices=["verbatim", "intended"])
    parser.add_argument("--tag_count", type=int, default=5)
    parser.add_argument("--language", default="en", help="Whisper language code.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="predictions.jsonl")
    args = parser.parse_args()

    device = args.device or _resolve_device()
    print(f"Loading model from {args.model} on {device}...")
    model, processor, torch_device, dtype = _load_model(args.model, device)

    sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    prefix_ids = _decoder_prefix_with_language(processor, model, args.language)

    # Load cached predictions
    done_ids: Dict[str, Dict[str, str]] = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done_ids[str(obj["id"])] = obj
        print(f"Found {len(done_ids)} cached predictions in {args.output}")

    for ds_name in args.datasets:
        for split in args.splits:
            print(f"\nProcessing {ds_name} / {split}...")
            ds = _load_hf_dataset(ds_name, split)
            print(f"  {len(ds)} samples, columns: {ds.column_names}")

            # Probe whether the native audio decoder works
            use_raw = False
            try:
                _ = ds[0]["audio"]
            except Exception:
                use_raw = True
                print("  Native audio decoder unavailable; falling back to raw bytes + soundfile.")

            # Pre-cast once if using raw mode (avoids repeated casting)
            if use_raw:
                from datasets import Audio
                ds_raw = ds.cast_column("audio", Audio(decode=False))
            else:
                ds_raw = None

            # Collect sample IDs via the underlying Arrow table to avoid
            # triggering the broken torchcodec audio decoder.
            arrow_table = ds.data
            id_col_name = "id" if "id" in ds.column_names else ("ID" if "ID" in ds.column_names else None)
            if id_col_name:
                id_col = [str(v.as_py()) for v in arrow_table.column(id_col_name)]
            else:
                id_col = [str(i) for i in range(len(ds))]

            samples = []
            for idx in range(len(ds)):
                samples.append({"id": id_col[idx], "split": split, "dataset": ds_name, "_idx": idx})

            # Filter to pending
            needs_v = "verbatim" in args.modes
            needs_i = "intended" in args.modes
            pending = []
            for s in samples:
                cached = done_ids.get(s["id"], {})
                nv = needs_v and not cached.get("pred_verbatim")
                ni = needs_i and not cached.get("pred_intended")
                if nv or ni:
                    pending.append(s)

            if not pending:
                print(f"  All {len(samples)} samples already cached, skipping.")
                continue

            print(f"  {len(pending)} samples to transcribe...")

            for mode in args.modes:
                prompt_text = _build_prompt(mode, args.tag_count)
                prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False).input_ids if prompt_text else []
                field = "pred_verbatim" if mode == "verbatim" else "pred_intended"

                mode_pending = [s for s in pending if not done_ids.get(s["id"], {}).get(field)]
                if not mode_pending:
                    continue

                print(f"  Mode: {mode} ({len(mode_pending)} samples)...")
                bs = max(1, args.batch_size)
                t0 = time.time()

                for i in range(0, len(mode_pending), bs):
                    batch = mode_pending[i:i + bs]
                    audio_arrays = []
                    for s in batch:
                        idx = s["_idx"]
                        if ds_raw is not None:
                            audio_field = ds_raw[idx]["audio"]
                        else:
                            audio_field = ds[idx]["audio"]
                        audio_arrays.append(_extract_audio_array(audio_field, sr))

                    texts = _transcribe_batch(
                        model, processor, torch_device, dtype,
                        audio_arrays, sr, prompt_ids, prefix_ids,
                        args.max_new_tokens, args.num_beams,
                    )

                    for s, text in zip(batch, texts):
                        sid = s["id"]
                        if sid not in done_ids:
                            done_ids[sid] = {
                                "id": sid,
                                "dataset": s["dataset"],
                                "split": s["split"],
                                "language": args.language,
                                "model": args.model,
                            }
                        done_ids[sid][field] = text

                    done_count = i + len(batch)
                    if done_count % max(50, bs) == 0 or done_count == len(mode_pending):
                        elapsed = time.time() - t0
                        print(f"    {done_count}/{len(mode_pending)} ({elapsed:.0f}s)")

    # Write all predictions
    print(f"\nWriting {len(done_ids)} predictions to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for obj in done_ids.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
