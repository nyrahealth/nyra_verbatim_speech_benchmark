"""
Normalization utilities for verbatim ASR evaluation.

Three normalization contexts with different rules:

1. Cutoff marker normalization:
   Converts between dataset convention (*) and model convention (-).
   Applied once during gold preprocessing before tokenization.

2. Token normalization (for alignment):
   Lowercase, preserve brackets and cutoff markers, strip trailing punctuation.
   Used when aligning sequences token-by-token.

3. WER/CER normalization:
   Aggressive — lowercase, remove all punctuation except brackets,
   collapse whitespace. Cutoff markers are stripped entirely.
   Used only for WER and CER computation.
"""

from __future__ import annotations

import re
import string as _string
from typing import List, Optional


# ── Cutoff Marker Normalization ────────────────────────────────────────────────

def normalize_cutoff_markers(
    text: str,
    source: str = "*",
    target: str = "-",
) -> str:
    """
    Convert cutoff marker convention in *text*.

    Datasets typically use ``*`` (e.g. ``th*``) while the model outputs ``-``
    (e.g. ``th-``).  This function normalises one convention to the other so
    that downstream processing only sees a single marker.

    Args:
        text:   Input text.
        source: The marker character to replace (default ``*``).
        target: The replacement character (default ``-``).
    """
    if not text:
        return text or ""
    pattern = rf"(\w)\{source}(?=\s|$|[.,!?;:'\"])"
    return re.sub(pattern, rf"\1{target}", text)


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Whitespace-split tokenization.  Punctuation stays attached to words."""
    return (text or "").split()


# ── Token Normalization (for alignment) ───────────────────────────────────────

def normalize_token(token: str) -> str:
    """
    Normalize a single token for alignment purposes.

    Rules:
      - Bracketed events are lowercased but kept intact: ``[UH]`` → ``[uh]``
      - Cutoff markers are preserved: ``th-`` → ``th-``
      - Everything else: lowercase, strip trailing ``.,!?;:"'-)``
        and leading ``("'``
    """
    if token.startswith("[") and token.endswith("]"):
        return token.lower()

    if token.rstrip(".,!?;:'\"").endswith("-"):
        return token.lower()

    normalized = token.lower()
    normalized = normalized.rstrip(".,!?;:\"'-)")
    normalized = normalized.lstrip("(\"'")
    return normalized


def normalize_tokens(tokens: List[str]) -> List[str]:
    """Apply :func:`normalize_token` to every element."""
    return [normalize_token(t) for t in tokens]


# ── WER / CER Normalization ──────────────────────────────────────────────────

_WER_KEEP = set("[]<>|_")
_WER_REMOVE = {c for c in _string.punctuation if c not in _WER_KEEP}


def normalize_for_wer(text: str) -> str:
    """
    Normalize text for WER / CER computation.

    - Lowercase
    - Remove all punctuation **except** ``[]<>|_``
    - Collapse whitespace

    Cutoff markers like ``w-`` become ``w`` because the hyphen is stripped.
    Bracket tokens like ``[laughter]`` survive because brackets are kept.
    """
    out = "".join(ch if ch not in _WER_REMOVE else " " for ch in (text or ""))
    return " ".join(out.split()).lower()


# ── Classification Helpers ────────────────────────────────────────────────────

FILLER_LABELS = frozenset({
    "[uh]", "[um]", "[er]", "[ah]", "[hm]", "[mm]", "[mhm]", "[erm]",
    "uh", "um", "er", "ah", "hm", "mm", "mhm", "erm", "uhh", "umm",
})

SOUND_EVENT_LABELS = frozenset({
    "[laughter]", "[laugh]", "[cough]", "[breath]", "[sigh]", "[noise]",
    "[music]", "[applause]", "[crying]", "[sneeze]", "[clearing]",
    "[lipsmack]", "[throatclearing]", "[inaudible]", "[crosstalk]",
    "[sniff]", "[yawn]", "[scream]", "[fart]",
    "laughter", "laugh", "cough", "breath", "sigh", "noise",
})


def is_filler(token: str) -> bool:
    """Check whether *token* is a filled pause (``[UH]``, ``um``, etc.)."""
    return normalize_token(token).lower() in FILLER_LABELS


def is_sound_event(token: str) -> bool:
    """Check whether *token* is a non-speech sound event."""
    norm = normalize_token(token).lower()
    if norm in SOUND_EVENT_LABELS:
        return True
    if norm.startswith("[") and norm.endswith("]") and len(norm) > 2:
        inner = norm[1:-1]
        if inner not in {"uh", "um", "er", "ah", "hm", "mm", "mhm", "erm"}:
            return True
    return False


def is_cutoff(token: str) -> bool:
    """Check whether *token* is a truncated word fragment (ends with ``-``)."""
    return token.rstrip(".,!?;:'\"").endswith("-")


def get_event_label(token: str) -> Optional[str]:
    """
    Extract the canonical label from an event token.

    ``[laughter]`` → ``laughter``, ``[UH]`` → ``uh``, bare ``um`` → ``um``.
    Returns ``None`` for non-event tokens.
    """
    norm = normalize_token(token).lower()
    if norm.startswith("[") and norm.endswith("]") and len(norm) > 2:
        return norm[1:-1]
    if norm in FILLER_LABELS or norm in SOUND_EVENT_LABELS:
        return norm
    return None
