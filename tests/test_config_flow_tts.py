"""Unit tests for the TTS helper logic in config_flow.py."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.venice_ai import config_flow as helpers
from custom_components.venice_ai.const import (
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_VOICE,
)


class TestTTSModelInfoExtraction:
    """Tests for ``_extract_tts_model_info``."""

    def test_model_spec_voices_extracted(self) -> None:
        info = helpers._extract_tts_model_info(
            [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["bm_daniel", "am_liam"],
                    },
                }
            ]
        )
        assert set(info.keys()) == {"tts-kokoro"}
        assert info["tts-kokoro"].voices == ["bm_daniel", "am_liam"]
        assert info["tts-kokoro"].default_voice == "bm_daniel"

    def test_explicit_default_voice(self) -> None:
        info = helpers._extract_tts_model_info(
            [
                {
                    "id": "tts-kokoro",
                    "default_voice": "am_liam",
                    "model_spec": {"voices": ["bm_daniel", "am_liam"]},
                }
            ]
        )
        assert info["tts-kokoro"].default_voice == "am_liam"

    def test_legacy_voice_models_fallback(self) -> None:
        info = helpers._extract_tts_model_info(
            [
                {
                    "id": "tts-legacy",
                    "voice_models": ["voice_a", "voice_b"],
                }
            ]
        )
        assert info["tts-legacy"].voices == ["voice_a", "voice_b"]
        assert info["tts-legacy"].default_voice == "voice_a"

    def test_models_without_voices_are_ignored(self) -> None:
        info = helpers._extract_tts_model_info(
            [
                {"id": "tts-empty", "model_spec": {"voices": []}},
                {"id": "tts-missing"},
            ]
        )
        assert not info


class TestParseCombinedTTSValue:
    """Tests for ``_parse_combined_tts_value``."""

    def test_parses_valid_value(self) -> None:
        assert helpers._parse_combined_tts_value("tts-kokoro → bm_daniel") == (
            "tts-kokoro",
            "bm_daniel",
        )

    def test_returns_none_for_invalid_value(self) -> None:
        assert helpers._parse_combined_tts_value("no-separator") is None
        assert helpers._parse_combined_tts_value(" → voice") is None
        assert helpers._parse_combined_tts_value("model → ") is None


class TestResolveCombinedTTSValue:
    """Tests for ``_resolve_combined_tts_value``."""

    @pytest.fixture
    def tts_info(self) -> dict[str, Any]:
        return {
            "tts-kokoro": helpers._TTSModelInfo(
                "tts-kokoro", ["bm_daniel", "am_liam"], "bm_daniel"
            ),
            "tts-eleven": helpers._TTSModelInfo("tts-eleven", ["rachel"], "rachel"),
        }

    def test_submitted_value_takes_priority(self, tts_info: dict[str, Any]) -> None:
        result = helpers._resolve_combined_tts_value(
            tts_info,
            {"tts_model_voice": "tts-eleven → rachel"},
            {CONF_TTS_MODEL: "tts-kokoro", CONF_TTS_VOICE: "bm_daniel"},
        )
        assert result == "tts-eleven → rachel"

    def test_saved_options_used_when_no_submission(
        self, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_combined_tts_value(
            tts_info,
            None,
            {CONF_TTS_MODEL: "tts-eleven", CONF_TTS_VOICE: "rachel"},
        )
        assert result == "tts-eleven → rachel"

    def test_recommended_default_fallback(self, tts_info: dict[str, Any]) -> None:
        result = helpers._resolve_combined_tts_value(tts_info, None, {})
        assert result == f"{RECOMMENDED_TTS_MODEL} → {RECOMMENDED_TTS_VOICE}"

    def test_first_model_when_recommended_missing(self) -> None:
        info = {"tts-other": helpers._TTSModelInfo("tts-other", ["v1"], "v1")}
        result = helpers._resolve_combined_tts_value(info, None, {})
        assert result == "tts-other → v1"

    def test_invalid_submission_falls_back_to_saved(
        self, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_combined_tts_value(
            tts_info,
            {"tts_model_voice": "invalid-value"},
            {CONF_TTS_MODEL: "tts-kokoro", CONF_TTS_VOICE: "bm_daniel"},
        )
        assert result == "tts-kokoro → bm_daniel"


class TestBuildCombinedTTSOptions:
    """Tests for ``_build_combined_tts_options``."""

    def test_builds_options_for_all_models(self) -> None:
        tts_info = {
            "tts-kokoro": helpers._TTSModelInfo(
                "tts-kokoro", ["bm_daniel", "am_liam"], "bm_daniel"
            ),
            "tts-eleven": helpers._TTSModelInfo("tts-eleven", ["rachel"], "rachel"),
        }
        options = helpers._build_combined_tts_options(tts_info)
        values = [o["value"] for o in options]
        assert values == [
            "tts-eleven → rachel",
            "tts-kokoro → bm_daniel",
            "tts-kokoro → am_liam",
        ]

    def test_empty_info_returns_recommended_fallback(self) -> None:
        options = helpers._build_combined_tts_options({})
        assert len(options) == 1
        assert (
            options[0]["value"] == f"{RECOMMENDED_TTS_MODEL} → {RECOMMENDED_TTS_VOICE}"
        )
