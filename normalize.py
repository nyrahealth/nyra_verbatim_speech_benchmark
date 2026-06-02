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

try:
    from num2words import num2words
except ImportError:  # pragma: no cover - optional dependency at import time
    num2words = None


# ── Cutoff Marker Normalization ─────────────────────────────────────────────

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


# ── Tokenization ───────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Whitespace-split tokenization with one extra cleanup:

    adjacent bracket tags like ``[UH][cough]`` are split into separate tokens
    before whitespace tokenization so they are treated as two events rather than
    one malformed combined token.
    """
    text = re.sub(r"(?<=\])(?=\[)", " ", text or "")
    return text.split()


# ── Token Normalization (for alignment) ────────────────────────────────────

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


# ── Classification Helpers ─────────────────────────────────────────────────

FILLER_LABELS = frozenset({
    "[uh]", "[um]", "[er]", "[ah]", "[hm]", "[mm]", "[mhm]", "[erm]",
    "uh", "um", "ah", "hm", "mm", "mhm", "erm", "uhh", "umm",
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


# ── Prediction Normalization ───────────────────────────────────────────────

_SOUND_ALIASES = {
    # Laughter
    (
        r"\[(?:laugh|laughs|laughing|chuckle|chuckles|"
        r"chuckling|giggles|giggling)\]"
    ): "[laughter]",
    r"\[(?:lacht|lachen)\]": "[laughter]",
    # Throat clearing
    (
        r"\[(?:clears?[ _-]throat|clearing[ _-]throat|"
        r"throat[ _-]clear(?:ing)?|clear[ _-]throat)\]"
    ): "[throatclearing]",
    r"\[räuspert sich\]": "[throatclearing]",
    # Cough
    r"\[(?:coughs|coughing)\]": "[cough]",
    r"\[hustet\]": "[cough]",
    # Breath / gasp
    (
        r"\[(?:breathe|breathing|breaths|inhales|exhales|"
        r"gasp|gasping)\]"
    ): "[breath]",
    r"\[atmet\]": "[breath]",
    # Sigh
    r"\[(?:sighs|sighing)\]": "[sigh]",
    r"\[seufzt\]": "[sigh]",
    # Sniff
    r"\[(?:sniffs|sniffing)\]": "[sniff]",
    r"\[schnieft\]": "[sniff]",
    # Lip smack
    r"\[lip[ _-]smacks?\]": "[lipsmack]",
    r"\[(?:schluckt|schmatzt)\]": "[lipsmack]",
    # Yawn
    r"\[gähnt\]": "[yawn]",
}

# Non-ASCII dash/minus variants. ASCII "-" is already the canonical marker and
# must not be rewritten before another word, otherwise compounds like
# "day-care" would become "day- care".
_CUTOFF_DASH_CHARS = r"‐‑‒–—―−﹘﹣－"


def normalize_prediction_cutoffs(text: str) -> str:
    """Normalize common model cutoff markers to the benchmark ``-`` marker."""
    text = normalize_cutoff_markers(text, source="*", target="-")
    # APIs may emit em/en dashes, Unicode hyphens, or minus signs for cutoffs.
    # Only normalize dash-like characters adjacent to word fragments. This
    # avoids turning ordinary spaced punctuation dashes into cutoff markers.
    dash = _CUTOFF_DASH_CHARS
    text = re.sub(rf"(\w)[{dash}](?=\s|$|[.,!?;:'\"])", r"\1-", text)
    text = re.sub(rf"(\w)[{dash}](?=\w)", r"\1- ", text)
    # ASCII hyphen is common in ordinary compounds, so only split very short
    # lowercase fragments like "w-well" or German "d-die".
    text = re.sub(r"\b([a-zäöüß]{1,2})-(?=\w)", r"\1- ", text)
    # Handle very short spaced fragments such as "d — die" conservatively.
    text = re.sub(rf"\b(\w{{1,2}})\s+[{dash}]\s+(?=\w)", r"\1- ", text)
    return text


def normalize_sound_aliases(text: str, language: str = "en") -> str:
    """Map bracketed sound aliases to the benchmark's canonical sound tags."""
    del language  # currently aliases are language-independent regexes.
    out = text or ""
    for pattern, replacement in _SOUND_ALIASES.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def normalize_fillers(text: str, language: str = "en") -> str:
    """Map safe bare/bracketed filled pauses to canonical filler tags."""
    out = text or ""

    # Bracketed variants are always safe to canonicalize.
    out = re.sub(r"\[uh\]", "[UH]", out, flags=re.IGNORECASE)
    out = re.sub(r"\[um\]", "[UM]", out, flags=re.IGNORECASE)
    out = re.sub(r"\[(?:äh|aeh)\]", "[UH]", out, flags=re.IGNORECASE)
    out = re.sub(r"\[(?:ähm|aehm)\]", "[UM]", out, flags=re.IGNORECASE)

    # Bare forms. Keep German conservative: bare "um" is a common real word.
    out = re.sub(r"\b[Uu]h[- ]huh\b", "[UH]", out)
    if language.lower().startswith("en"):
        out = re.sub(r"\b[Uu]h\b", "[UH]", out)
        out = re.sub(r"\b[Uu]m+\b", "[UM]", out)

    out = re.sub(r"\b(?:[Ää]h|aeh)\b", "[UH]", out)
    out = re.sub(r"\b(?:[Ää]hm|aehm)\b", "[UM]", out)
    out = re.sub(r"\b[Hh]mm?\b", "[UM]", out)
    out = re.sub(r"\b[Mm]hm\b", "[UM]", out)
    out = re.sub(r"\b[Mm]m\b", "[UM]", out)
    return out


def normalize_numbers(text: str, language: str = "en") -> str:
    """Convert simple digit forms when ``num2words`` is installed."""
    if not text or num2words is None:
        return text or ""

    lang = "de" if language.lower().startswith("de") else "en"

    def _safe_num(word: str, *, to: Optional[str] = None) -> Optional[str]:
        try:
            value = int(word.replace(",", ""))
            if to:
                return num2words(value, to=to, lang=lang)
            return num2words(value, lang=lang)
        except (ValueError, TypeError, OverflowError, NotImplementedError):
            return None

    def _replace_currency(unit_en: str, unit_de: str):
        unit = unit_de if lang == "de" else unit_en

        def repl(match: re.Match) -> str:
            spoken = _safe_num(match.group(1))
            return f"{spoken} {unit}" if spoken else match.group(0)

        return repl

    def _replace_percent(match: re.Match) -> str:
        spoken = _safe_num(match.group(1))
        unit = "Prozent" if lang == "de" else "percent"
        return f"{spoken} {unit}" if spoken else match.group(0)

    def _replace_ordinal(match: re.Match) -> str:
        spoken = _safe_num(match.group(1), to="ordinal")
        return spoken if spoken else match.group(0)

    def _replace_number(match: re.Match) -> str:
        spoken = _safe_num(match.group(0))
        return spoken if spoken else match.group(0)

    out = text
    out = re.sub(r"\$(\d[\d,]*)", _replace_currency("dollars", "Dollar"), out)
    out = re.sub(r"€(\d[\d,]*)", _replace_currency("euros", "Euro"), out)
    out = re.sub(r"(\d[\d,]*)%", _replace_percent, out)
    if lang == "en":
        out = re.sub(
            r"\b(\d+)(?:st|nd|rd|th)\b",
            _replace_ordinal,
            out,
            flags=re.IGNORECASE,
        )
    out = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", _replace_number, out)
    out = re.sub(r"\b\d+\b", _replace_number, out)
    return out


def normalize_spacing(text: str) -> str:
    """Clean whitespace and separate adjacent bracket tags."""
    out = re.sub(r"(?<=\])(?=\[)", " ", text or "")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def normalize_prediction(text: str, language: str = "en") -> str:
    """
    Normalize model output into the canonical verbatim format.

    The function is conservative and idempotent: applying it to an already
    canonical transcript should not change its semantics.
    """
    out = text or ""
    out = normalize_prediction_cutoffs(out)
    out = normalize_sound_aliases(out, language)
    out = normalize_fillers(out, language)
    out = normalize_numbers(out, language)
    out = normalize_spacing(out)
    return out
