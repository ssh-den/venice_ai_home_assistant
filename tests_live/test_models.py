"""Checks of the Venice model lists; listing models is free."""

from __future__ import annotations

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.const import (
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TTS_MODEL,
)
from custom_components.venice_ai.languages import unlisted_models
from custom_components.venice_ai.models import parse_models, parse_tts_models


def _ids(models: list[dict]) -> set[str]:
    return {m["id"] for m in models}


async def test_recommended_models_exist(live_entry: MockConfigEntry) -> None:
    data = live_entry.runtime_data.coordinator.data
    assert RECOMMENDED_CHAT_MODEL in _ids(data["text_models"])
    assert RECOMMENDED_TTS_MODEL in _ids(data["tts_models"])
    assert RECOMMENDED_STT_MODEL in _ids(data["asr_models"])
    assert parse_models(data["text_models"])[
        RECOMMENDED_CHAT_MODEL
    ].supports_function_calling


async def test_language_table_covers_all_speech_models(
    live_entry: MockConfigEntry,
) -> None:
    data = live_entry.runtime_data.coordinator.data
    assert not unlisted_models(_ids(data["tts_models"]), _ids(data["asr_models"]))


async def test_speech_models_report_privacy_and_formats(
    live_entry: MockConfigEntry,
) -> None:
    data = live_entry.runtime_data.coordinator.data
    tts = parse_tts_models(data["tts_models"])
    assert tts, "no TTS model with voices"
    for model in tts.values():
        assert model.privacy, model.id
        assert model.formats, model.id
        assert model.audio_format in ("mp3", "wav"), model.id
    for model in data["asr_models"]:
        assert model.get("model_spec", {}).get("privacy"), model["id"]
