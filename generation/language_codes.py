"""Per-provider language code mappings for benchmark generation."""

from __future__ import annotations

# ISO 639-1 codes used by our synth corpora and HuggingFace datasets.
SYNTH_LANGUAGES = ("es", "nl", "fr", "it", "pl", "pt", "ru", "sv")
BENCHMARK_LANGUAGES = ("en", "de", *SYNTH_LANGUAGES)

# ElevenLabs Scribe expects ISO 639-3 (or similar 3-letter) codes.
ELEVENLABS_CODES = {
    "en": "eng",
    "de": "deu",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "nl": "nld",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "sv": "swe",
}

# Inworld STT expects BCP-47 locale tags.
INWORLD_CODES = {
    "en": "en-US",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "sv": "sv-SE",
}

# MAI-Transcribe accepts BCP-47 locale tags in the locales field.
MAI_LOCALES = {
    "en": "en-US",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "sv": "sv-SE",
}


def iso_639_1(language: str) -> str:
    """Return the benchmark ISO 639-1 code unchanged."""
    return language


def assemblyai_code(language: str) -> str:
    return iso_639_1(language)


def assemblyai_speech_models(language: str) -> list[str]:
    del language
    return ["universal-3-pro"]


def elevenlabs_code(language: str) -> str:
    return ELEVENLABS_CODES.get(language, language)


def inworld_code(language: str) -> str:
    return INWORLD_CODES.get(language, language)


def mai_locale(language: str) -> str:
    if language not in MAI_LOCALES:
        raise KeyError(f"No MAI locale mapping for language: {language}")
    return MAI_LOCALES[language]
