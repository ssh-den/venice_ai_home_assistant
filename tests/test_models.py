"""Tests for parsing Venice AI model capabilities."""

from __future__ import annotations

import pytest

from custom_components.venice_ai.languages import MULTILINGUAL
from custom_components.venice_ai.models import (
    ModelInfo,
    TTSModel,
    find_model,
    parse_models,
    parse_tts_models,
)

TEE_MODEL = {
    "id": "tee-model",
    "model_spec": {
        "name": "TEE Model",
        "privacy": "private",
        "offline": True,
        "capabilities": {
            "supportsFunctionCalling": True,
            "supportsResponseSchema": True,
            "supportsTeeAttestation": True,
            "supportsE2EE": False,
        },
        "pricing": {"input": {"usd": 0.5}, "output": {"usd": 2}},
    },
}


def test_from_api() -> None:
    info = ModelInfo.from_api(TEE_MODEL)
    assert info == ModelInfo(
        id="tee-model",
        name="TEE Model",
        privacy="private",
        supports_function_calling=True,
        supports_response_schema=True,
        supports_tee=True,
        supports_e2ee=False,
        input_price=0.5,
        output_price=2.0,
        offline=True,
    )
    assert info.label == "tee-model (TEE, tools, $0.5/$2 per 1M, offline)"


def test_from_api_without_spec() -> None:
    info = ModelInfo.from_api({"id": "plain", "model_spec": "bogus"})
    assert info == ModelInfo(id="plain", name="plain")
    assert info.label == "plain"


def test_privacy_label() -> None:
    info = ModelInfo.from_api(
        {"id": "anon", "model_spec": {"privacy": "anonymized", "pricing": {}}}
    )
    assert info.label == "anon (Anonymized)"


def test_parse_models_skips_invalid_entries() -> None:
    models = parse_models([TEE_MODEL, "junk", {"name": "no id"}])
    assert list(models) == ["tee-model"]


def test_find_model() -> None:
    assert find_model([TEE_MODEL], "tee-model") is not None
    assert find_model([TEE_MODEL], "other") is None
    assert find_model(None, "tee-model") is None


def test_tts_model_from_api() -> None:
    model = TTSModel.from_api(
        {
            "id": "tts-kokoro",
            "model_spec": {
                "voices": ["af_heart", "zf_xiaobei", "pm_alex"],
                "supported_formats": ["mp3", "wav"],
                "default_format": "mp3",
            },
        }
    )
    assert model.languages == ["en", "pt", "zh"]
    assert model.voices_for("zh-CN") == ["zf_xiaobei"]
    assert model.voices_for("ru") == ["af_heart", "zf_xiaobei", "pm_alex"]
    assert model.audio_format == "mp3"


def test_tts_model_multilingual_voices() -> None:
    model = TTSModel("tts-xai", ("eve", "ara"), ("wav", "pcm"), "wav")
    assert model.languages == list(MULTILINGUAL)
    assert model.voices_for("ru") == ["eve", "ara"]
    assert model.audio_format == "wav"


@pytest.mark.parametrize(
    ("formats", "default", "expected"),
    [((), None, "mp3"), (("opus", "pcm"), "opus", "opus"), (("flac",), None, "flac")],
)
def test_tts_audio_format(
    formats: tuple[str, ...], default: str | None, expected: str
) -> None:
    assert TTSModel("m", ("v",), formats, default).audio_format == expected


def test_parse_tts_models() -> None:
    models = parse_tts_models(
        [
            {"id": "spec", "model_spec": {"voices": ["a"]}},
            {"id": "legacy", "voice_models": ["b"]},
            {"id": "silent", "model_spec": {"voices": []}},
            "junk",
        ]
    )
    assert {k: v.voices for k, v in models.items()} == {
        "spec": ("a",),
        "legacy": ("b",),
    }
