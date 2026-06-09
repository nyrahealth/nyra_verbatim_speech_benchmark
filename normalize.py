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


def is_filler(token: str, language: str = "en") -> bool:
    """Check whether *token* is a filled pause (``[UH]``, ``um``, etc.)."""
    norm = normalize_token(token).lower()
    lang = _primary_language_code(language)
    if lang == "de" and norm == "um":
        return False
    return norm in FILLER_LABELS


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


def get_event_label(token: str, language: str = "en") -> Optional[str]:
    """
    Extract the canonical label from an event token.

    ``[laughter]`` → ``laughter``, ``[UH]`` → ``uh``, bare ``um`` → ``um``.
    Returns ``None`` for non-event tokens.
    """
    norm = normalize_token(token).lower()
    if norm.startswith("[") and norm.endswith("]") and len(norm) > 2:
        return norm[1:-1]
    if is_filler(norm, language=language) or norm in SOUND_EVENT_LABELS:
        return norm
    return None


# ── Prediction Normalization ───────────────────────────────────────────────

_SYNTH_LANGUAGES = frozenset({"es", "nl", "fr", "it", "pl", "pt", "ru", "sv"})

_NUM2WORDS_LANGUAGES = frozenset(_SYNTH_LANGUAGES | {"en", "de"})


def _primary_language_code(language: str) -> str:
    return (language or "en").lower().split("-")[0]


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

# Observed in multilingual API outputs. Applied only for the matching language
# so the existing English/German alias table above stays unchanged.
_LANG_SOUND_ALIASES = {
    "es": [
        (r"\[carraspe[ao]\]", "[throatclearing]"),
        (r"\[(?:risa|risas|ríe|sonidos de risa)\]", "[laughter]"),
        (r"\[(?:tos|tose)\]", "[cough]"),
        (r"\[suspiro\]", "[sigh]"),
        (r"\[exhala\]", "[breath]"),
        (r"\[bostezo\]", "[yawn]"),
        (r"\[chasquido de lengua\]", "[lipsmack]"),
    ],
    "nl": [
        (r"\[schraapt keel\]", "[throatclearing]"),
        (r"\[(?:kucht|hoest)\]", "[cough]"),
        (r"\[gniffelt\]", "[laughter]"),
        (r"\[zucht\]", "[sigh]"),
        (r"\[klikt met tong\]", "[lipsmack]"),
    ],
    "fr": [
        (r"\[raclement de gorge\]", "[throatclearing]"),
        (r"\[(?:rire|rires)\]", "[laughter]"),
        (r"\[(?:tousse|toux)\]", "[cough]"),
        (r"\[bâillement\]", "[yawn]"),
        (r"\[claquement de langue\]", "[lipsmack]"),
        (r"\[soupir\]", "[sigh]"),
    ],
    "it": [
        (r"\[(?:schiarisce la voce|si schiarisce la gola)\]", "[throatclearing]"),
        (r"\[(?:risatina|ride|risata)\]", "[laughter]"),
        (r"\[tossisce\]", "[cough]"),
        (
            r"\[(?:schiocco(?: di lingua)?|schiocca la lingua)\]",
            "[lipsmack]",
        ),
        (r"\[sbuffa\]", "[sigh]"),
    ],
    "pl": [
        (r"\[(?:chrząknięcie|odchrząknięcie)\]", "[throatclearing]"),
        (r"\[śmiech\]", "[laughter]"),
        (r"\[(?:ziewa|ziewnięcie)\]", "[yawn]"),
        (r"\[(?:kaszle|kaszlnięcie|kaszel)\]", "[cough]"),
        (r"\[westchnięcie\]", "[sigh]"),
    ],
    "pt": [
        (r"\[pigarro\]", "[throatclearing]"),
        (r"\[(?:riso|risada|chuckle)\]", "[laughter]"),
        (r"\[tosse\]", "[cough]"),
        (r"\[suspiro\]", "[sigh]"),
        (r"\[estalo de língua\]", "[lipsmack]"),
        (r"\[bocejo\]", "[yawn]"),
    ],
    "ru": [
        (r"\[прочищает горло\]", "[throatclearing]"),
        (r"\[(?:смешок|смеется)\]", "[laughter]"),
        (r"\[(?:кашляет|кашель)\]", "[cough]"),
        (r"\[вздыхает\]", "[sigh]"),
        (r"\[цокает\]", "[lipsmack]"),
    ],
    "sv": [
        (r"\[(?:harklar sig|harkling)\]", "[throatclearing]"),
        (r"\[(?:skratt|skrattar)\]", "[laughter]"),
        (r"\[(?:hostar|hosta)\]", "[cough]"),
        (r"\[(?:suck|suckar)\]", "[sigh]"),
        (r"\[(?:smackar med munnen|smack|klickljud)\]", "[lipsmack]"),
        (r"\[gäspar\]", "[yawn]"),
    ],
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
    out = text or ""
    for pattern, replacement in _SOUND_ALIASES.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    lang = _primary_language_code(language)
    for pattern, replacement in _LANG_SOUND_ALIASES.get(lang, ()):
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _normalize_fillers_multilingual(text: str, language: str) -> str:
    """Map language-specific filled pauses observed in synth API outputs."""
    out = text or ""
    lang = _primary_language_code(language)
    if lang not in _SYNTH_LANGUAGES:
        return out

    for pattern, replacement in (
        (r"\[eh\]", "[UH]"),
        (r"\[euh\]", "[UH]"),
        (r"\[ehm\]", "[UM]"),
        (r"\[emm\]", "[UM]"),
        (r"\[öh\]", "[UH]"),
    ):
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    # Drawn-out "mmm" hesitation is unambiguous across the synth languages.
    # (English/German keep their existing 1-2 char "mm" rule untouched.)
    out = re.sub(r"\bm{3,}\b", "[UM]", out, flags=re.IGNORECASE)

    # Cyrillic filled pauses sometimes leak into Latin-script transcripts. The
    # characters never form real words there, so map them for every synth lang.
    out = re.sub(r"\bэ+м+\b", "[UM]", out, flags=re.IGNORECASE)
    out = re.sub(r"\bэ+\b", "[UH]", out, flags=re.IGNORECASE)

    if lang == "es":
        out = re.sub(r"\b[Ee]h+\b", "[UH]", out)
        out = re.sub(r"\b[Ee]hm+\b", "[UM]", out)
        out = re.sub(r"\b[Ee]mm*\b", "[UM]", out)
    elif lang == "nl":
        out = re.sub(r"\b[Ee]uhm\b", "[UM]", out)
        out = re.sub(r"\b[Ee]h\b", "[UH]", out)
        out = re.sub(r"\b[Uu]h\b", "[UH]", out)
        out = re.sub(r"\b(?:[Ee]hm|[Uu]hm)\b", "[UM]", out)
        out = re.sub(r"\b[Ee]uh\b", "[UH]", out)
    elif lang == "fr":
        out = re.sub(r"\b[Ee]uhm\b", "[UM]", out)
        out = re.sub(r"\b[Ee]uh+\b", "[UH]", out)
        out = re.sub(r"\b[Ee]hm+\b", "[UM]", out)
        out = re.sub(r"\b[Bb]ah\b", "[UH]", out)
    elif lang == "it":
        out = re.sub(r"\b[Ee]hm+\b", "[UM]", out)
        out = re.sub(r"\b[Ee]h+\b", "[UH]", out)
        out = re.sub(r"\b[Ee]m\b", "[UM]", out)
    elif lang == "pl":
        out = re.sub(r"\b[Ee]hm+\b", "[UM]", out)
        out = re.sub(r"\b[Ee]h+\b", "[UH]", out)
        # Drawn-out vowel hesitations written as repeated letters ("yyy", "eee").
        # Require 2+ repeats so single-letter words ("a", "e") are untouched.
        out = re.sub(r"\by{2,}\b", "[UM]", out, flags=re.IGNORECASE)
        out = re.sub(r"\be{2,}\b", "[UH]", out, flags=re.IGNORECASE)
        # Some models transcribe the hesitation in Cyrillic.
        out = re.sub(r"\bэ+м+\b", "[UM]", out, flags=re.IGNORECASE)
        out = re.sub(r"\bэ+\b", "[UH]", out, flags=re.IGNORECASE)
    elif lang == "ru":
        # Cyrillic filled pauses: "э"/"ээ" → uh, "эм"/"эмм" → um.
        # "ну", "это", "вот" are real discourse words and are left untouched.
        out = re.sub(r"\bэ+м+\b", "[UM]", out, flags=re.IGNORECASE)
        out = re.sub(r"\bэ+\b", "[UH]", out, flags=re.IGNORECASE)
        out = re.sub(r"\bм{2,}\b", "[UM]", out, flags=re.IGNORECASE)
    elif lang == "pt":
        out = re.sub(r"\b[Ee]h+\b", "[UH]", out)
        out = re.sub(r"\b[Ee]hm+\b", "[UM]", out)
    elif lang == "sv":
        out = re.sub(r"\b[Ee]h+\b", "[UH]", out)
        out = re.sub(r"\b[Ee]hm+\b", "[UM]", out)
        out = re.sub(r"\b[Öö]h\b", "[UH]", out)

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
    out = _normalize_fillers_multilingual(out, language)
    return out


def _num2words_language(language: str) -> str:
    lang = _primary_language_code(language)
    if lang.startswith("de"):
        return "de"
    if lang in _NUM2WORDS_LANGUAGES:
        return lang
    return "en"


def normalize_numbers(text: str, language: str = "en") -> str:
    """Convert simple digit forms when ``num2words`` is installed."""
    if not text or num2words is None:
        return text or ""

    lang = _num2words_language(language)

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
