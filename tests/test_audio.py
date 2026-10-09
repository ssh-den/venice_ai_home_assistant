"""Tests for the speech audio and text helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from custom_components.venice_ai.audio import (
    WavStreamJoiner,
    async_stream_segments,
    join_wav,
    pack_sentences,
    pcm_to_wav,
    split_text,
    wav_parts,
)


async def _chunks(*parts: str) -> AsyncGenerator[str]:
    for part in parts:
        yield part


def test_pcm_to_wav_header() -> None:
    wav = pcm_to_wav(b"\x01\x02\x03\x04")
    assert wav[:4] == b"RIFF"
    assert wav[8:16] == b"WAVEfmt "
    assert len(wav) == 48
    assert wav_parts(wav)[1] == b"\x01\x02\x03\x04"


def test_split_text_respects_limit() -> None:
    assert split_text("One. Two. Three.", limit=10) == ["One. Two.", "Three."]
    assert split_text("x" * 25, limit=10) == ["x" * 10, "x" * 10, "x" * 5]
    assert not split_text("")


def test_pack_sentences_splits_long_sentences_at_words() -> None:
    assert pack_sentences(["aaa bbb ccc"], limit=7) == ["aaa bbb", "ccc"]


async def test_stream_segments_yields_first_sentence_alone() -> None:
    segments = [
        s async for s in async_stream_segments(_chunks("Hi there. ", "More. ", "End"))
    ]
    assert segments[0] == "Hi there."
    assert " ".join(segments) == "Hi there. More. End"


async def test_stream_segments_propagates_errors() -> None:
    async def _broken() -> AsyncGenerator[str]:
        yield "Hi. "
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _ = [s async for s in async_stream_segments(_broken())]


def test_join_wav() -> None:
    joined = join_wav([pcm_to_wav(b"\x01\x02"), pcm_to_wav(b"\x03\x04")])
    assert joined == pcm_to_wav(b"\x01\x02\x03\x04")


def test_wav_stream_joiner() -> None:
    joiner = WavStreamJoiner()
    first = joiner.add(pcm_to_wav(b"\x01\x02"))
    second = joiner.add(pcm_to_wav(b"\x03\x04"))
    assert first.startswith(b"RIFF\xff\xff\xff\xff")
    assert first.endswith(b"\xff\xff\xff\xff\x01\x02")
    assert second == b"\x03\x04"


@pytest.mark.parametrize("data", [b"junk", b"RIFF\x00\x00\x00\x00WAVEfmt "])
def test_wav_parts_rejects_invalid_data(data: bytes) -> None:
    with pytest.raises(ValueError):
        wav_parts(data)
