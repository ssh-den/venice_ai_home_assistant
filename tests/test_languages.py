"""Tests for the speech model language tables."""

from __future__ import annotations

import pytest

from custom_components.venice_ai.languages import (
    LANGUAGE_NAMES,
    MULTILINGUAL,
    STT_MODELS,
    TTS_MODELS,
    Hint,
    primary_language,
    stt_languages,
    tts_language_hint,
    tts_languages,
    tts_voices,
    unlisted_models,
)

TABLES = [
    *(info.languages for info in TTS_MODELS.values()),
    *STT_MODELS.values(),
    MULTILINGUAL,
]


@pytest.mark.parametrize("languages", TABLES)
def test_tables_hold_unique_lowercase_codes(languages: tuple[str, ...]) -> None:
    assert languages
    assert len(set(languages)) == len(languages)
    assert all(code == primary_language(code) for code in languages)


@pytest.mark.parametrize(
    "model", [m for m, info in TTS_MODELS.items() if info.hint is Hint.NAME]
)
def test_name_hints_cover_every_language(model: str) -> None:
    assert set(TTS_MODELS[model].languages) <= set(LANGUAGE_NAMES)


def test_primary_language() -> None:
    assert primary_language("en-US") == "en"
    assert primary_language("pt_BR") == "pt"
    assert primary_language("yue") == "yue"


def test_stt_languages() -> None:
    parakeet = stt_languages("nvidia/parakeet-tdt-0.6b-v3")
    assert "ru" in parakeet
    assert "ja" not in parakeet
    assert "ja" in stt_languages("openai/whisper-large-v3")
    assert stt_languages("unknown") == list(MULTILINGUAL)


def test_tts_languages() -> None:
    assert tts_languages("tts-orpheus", ("tara",)) == ["en"]
    assert tts_languages("tts-kokoro", ("af_heart", "jf_alpha")) == ["en", "ja"]
    assert "ru" in tts_languages("tts-xai-v1", ("eve",))
    assert tts_languages("unknown", ("v",)) == list(MULTILINGUAL)


def test_tts_voices() -> None:
    voices = ("af_heart", "jf_alpha")
    assert tts_voices("tts-kokoro", voices, "ja-JP") == ["jf_alpha"]
    assert tts_voices("tts-kokoro", voices, "ru") == list(voices)
    assert tts_voices("tts-xai-v1", ("eve", "ara"), "ja") == ["eve", "ara"]


@pytest.mark.parametrize(
    ("model", "language", "expected"),
    [
        ("tts-xai-v1", "pt-BR", "pt"),
        ("tts-elevenlabs-turbo-v2-5", "nb", "no"),
        ("tts-minimax-speech-02-hd", "yue", "Chinese,Yue"),
        ("tts-qwen3-0-6b", "uk", None),
        ("tts-kokoro", "en", None),
        ("unknown", "en", None),
    ],
)
def test_tts_language_hint(model: str, language: str, expected: str | None) -> None:
    assert tts_language_hint(model, language) == expected


def test_unlisted_models() -> None:
    assert unlisted_models(["tts-kokoro", "tts-new"], ["stt-new", "fal-ai/wizper"]) == [
        "stt-new",
        "tts-new",
    ]
