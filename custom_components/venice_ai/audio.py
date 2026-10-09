"""Audio and text helpers for the speech platforms."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Iterable
import struct

from sentence_stream import SentenceBoundaryDetector, stream_to_sentences

# Venice rejects speech requests with more input characters than this.
TTS_INPUT_LIMIT = 4096

# Size fields of a WAV stream whose length is not known in advance.
_UNKNOWN_SIZE = 0xFFFFFFFF


def _split_long(sentence: str, limit: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > limit:
            parts.append(current)
            candidate = word
        current = candidate
    if current:
        parts.append(current)
    return parts


def pack_sentences(sentences: Iterable[str], limit: int = TTS_INPUT_LIMIT) -> list[str]:
    """Join sentences into as few request inputs of at most ``limit`` as possible."""
    segments: list[str] = []
    current = ""
    for sentence in sentences:
        for part in _split_long(sentence.strip(), limit):
            candidate = f"{current} {part}" if current else part
            if len(candidate) > limit:
                segments.append(current)
                candidate = part
            current = candidate
    if current:
        segments.append(current)
    return segments


def split_text(text: str, limit: int = TTS_INPUT_LIMIT) -> list[str]:
    """Split a complete text into request inputs at sentence boundaries."""
    return pack_sentences(stream_to_sentences([text]), limit)


async def async_stream_segments(
    text_stream: AsyncIterable[str], limit: int = TTS_INPUT_LIMIT
) -> AsyncGenerator[str]:
    """Group streamed text into request inputs.

    The first sentence is yielded on its own so speech starts early; after
    that every sentence completed in the meantime is sent in one request.
    """
    sentences: list[str] = []
    ready = asyncio.Event()
    done = False

    async def _read() -> None:
        nonlocal done
        detector = SentenceBoundaryDetector()
        try:
            async for chunk in text_stream:
                for sentence in detector.add_chunk(chunk):
                    if sentence.strip():
                        sentences.append(sentence)
                        ready.set()
            if (rest := detector.finish()).strip():
                sentences.append(rest)
        finally:
            done = True
            ready.set()

    reader = asyncio.create_task(_read())
    try:
        first = True
        while True:
            await ready.wait()
            if not done:
                ready.clear()
            if not sentences:
                if done:
                    break
                continue
            batch = sentences[:1] if first else sentences[:]
            del sentences[: len(batch)]
            first = False
            for segment in pack_sentences(batch, limit):
                yield segment
        await reader
    finally:
        if not reader.done():
            reader.cancel()


def _riff(fmt: bytes, data_size: int) -> bytes:
    riff_size = (
        _UNKNOWN_SIZE if data_size == _UNKNOWN_SIZE else 20 + len(fmt) + data_size
    )
    return (
        struct.pack("<4sL4s4sL", b"RIFF", riff_size, b"WAVE", b"fmt ", len(fmt))
        + fmt
        + struct.pack("<4sL", b"data", data_size)
    )


def pcm_to_wav(
    pcm: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16
) -> bytes:
    """Wrap raw PCM samples in a WAV container."""
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHLLHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    return _riff(fmt, len(pcm)) + pcm


def wav_parts(wav: bytes) -> tuple[bytes, bytes]:
    """Return the fmt chunk and the samples of a WAV file."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("Not a WAV file")
    fmt = b""
    offset = 12
    while offset + 8 <= len(wav):
        chunk_id, size = struct.unpack_from("<4sL", wav, offset)
        start = offset + 8
        if chunk_id == b"data":
            end = len(wav) if size in (0, _UNKNOWN_SIZE) else start + size
            return fmt, wav[start:end]
        if chunk_id == b"fmt ":
            fmt = wav[start : start + size]
        offset = start + size + (size & 1)
    raise ValueError("WAV file has no data chunk")


def join_wav(wavs: Iterable[bytes]) -> bytes:
    """Concatenate WAV files that share one audio format."""
    fmt = b""
    samples: list[bytes] = []
    for wav in wavs:
        chunk_fmt, data = wav_parts(wav)
        fmt = fmt or chunk_fmt
        samples.append(data)
    pcm = b"".join(samples)
    return _riff(fmt, len(pcm)) + pcm


class WavStreamJoiner:
    """Turn consecutive WAV files into one WAV stream of unknown length."""

    def __init__(self) -> None:
        """Initialize before the first file."""
        self._started = False

    def add(self, wav: bytes) -> bytes:
        """Return the bytes to send for the next WAV file."""
        fmt, data = wav_parts(wav)
        if self._started:
            return data
        self._started = True
        return _riff(fmt, _UNKNOWN_SIZE) + data
