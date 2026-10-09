"""Languages of the Venice AI speech models.

Venice does not report which languages a speech model supports, so this module
is the single source of truth for them. Each entry follows the model vendor's
documentation; update it when Venice adds or changes a speech model.

Codes are the language subtags Home Assistant uses (ISO 639-1 where one
exists). Models missing from the tables are treated as multilingual.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import re


def _codes(codes: str) -> tuple[str, ...]:
    return tuple(codes.split())


class Hint(StrEnum):
    """How Venice expects the ``language`` parameter of a TTS model."""

    CODE = "code"
    NAME = "name"


@dataclass(frozen=True, slots=True)
class TTSLanguages:
    """Languages of a text-to-speech model."""

    languages: tuple[str, ...]
    hint: Hint | None = None
    # Voices are language specific and named after their language
    per_voice: bool = False


# Whisper large-v3 (openai/whisper tokenizer), with the Home Assistant codes
# for Javanese (jv) and Norwegian (nb)
WHISPER = _codes("""
    af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr
    gl gu ha haw he hi hr ht hu hy id is it ja jv ka kk km kn ko la lb ln lo lt lv
    mg mi mk ml mn mr ms mt my nb ne nl nn oc pa pl ps pt ro ru sa sd si sk sl sn
    so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo yue zh
    """)

# Fallback for models that are not listed below
MULTILINGUAL = WHISPER

# Kokoro voice IDs start with a language letter and a gender letter
# (hexgrad/Kokoro-82M VOICES.md), e.g. jf_alpha is a Japanese female voice.
KOKORO_VOICE_LANGUAGES = {
    "a": "en",
    "b": "en",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt",
    "z": "zh",
}

_QWEN3_TTS = _codes("de en es fr it ja ko pt ru zh")

TTS_MODELS: dict[str, TTSLanguages] = {
    "tts-kokoro": TTSLanguages(
        tuple(dict.fromkeys(KOKORO_VOICE_LANGUAGES.values())), per_voice=True
    ),
    # Qwen3-TTS model card
    "tts-qwen3-0-6b": TTSLanguages(_QWEN3_TTS, Hint.NAME),
    "tts-qwen3-1-7b": TTSLanguages(_QWEN3_TTS, Hint.NAME),
    # docs.x.ai text-to-speech
    "tts-xai-v1": TTSLanguages(
        _codes("ar bn de en es fr hi id it ja ko pt ru tr vi zh"), Hint.CODE
    ),
    # docs.inworld.ai TTS models
    "tts-inworld-1-5-max": TTSLanguages(
        _codes("ar de en es fr he hi it ja ko nl pl pt ru zh")
    ),
    # Resemble Chatterbox HD on fal is English only
    "tts-chatterbox-hd": TTSLanguages(("en",)),
    # Canopy Labs Orpheus 3B is English only
    "tts-orpheus": TTSLanguages(("en",)),
    # ElevenLabs models overview: eleven_turbo_v2_5
    "tts-elevenlabs-turbo-v2-5": TTSLanguages(
        _codes("""
            ar bg cs da de el en es fi fil fr hi hr hu id it ja ko ms nb nl pl pt ro
            ru sk sv ta tr uk vi zh
            """),
        Hint.CODE,
    ),
    # MiniMax T2A language_boost values, minus Persian, Filipino and Tamil,
    # which the speech-02 series does not support
    "tts-minimax-speech-02-hd": TTSLanguages(
        _codes("""
            af ar bg ca cs da de el en es fi fr he hi hr hu id it ja ko ms nb nl nn
            pl pt ro ru sk sl sv th tr uk vi yue zh
            """),
        Hint.NAME,
    ),
    # Google Cloud Gemini-TTS language availability; the input language is
    # detected automatically
    "tts-gemini-3-1-flash": TTSLanguages(_codes("""
            af am ar az be bg bn ca ceb cs da de el en es et eu fa fi fil fr gl gu he
            hi hr ht hu hy id is it ja jv ka kn ko kok la lb lo lt lv mai mg mk ml mn
            mr ms my nb ne nl nn or pa pl ps pt ro ru sd si sk sl sq sr sv sw ta te th
            tr uk ur vi zh
            """)),
    # Gradium API documentation
    "tts-gradium-v1": TTSLanguages(_codes("de en es fr pt")),
}

STT_MODELS: dict[str, tuple[str, ...]] = {
    # nvidia/parakeet-tdt-0.6b-v3 model card
    "nvidia/parakeet-tdt-0.6b-v3": _codes(
        "bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk"
    ),
    "openai/whisper-large-v3": WHISPER,
    # Wizper is Whisper large-v3 served by fal
    "fal-ai/wizper": WHISPER,
    # ElevenLabs speech-to-text supported languages
    "elevenlabs/scribe-v2": _codes("""
        af am ar as ast az be bg bn bs ca ceb cs cy da de el en es et fa ff fi fil fr
        ga gl gu ha he hi hr hu hy id ig is it ja jv ka kea kk km kn ko ku ky lb lg
        ln lo lt luo lv mi mk ml mn mr ms mt my nb ne nl nso ny oc or pa pl ps pt ro
        ru sd si sk sl sn so sr sv sw ta te tg th tr uk umb ur uz vi wo xh yue zh zu
        """),
    # docs.x.ai speech-to-text
    "stt-xai-v1": _codes("""
        ar bg bs ca cs da de el en es fa fi fil fr hi hr hu id it ja ko mk ms nb nl
        pl pt ro ru sk sv th tr uk ur vi yue zh
        """),
}

# English names for Hint.NAME models (Qwen3-TTS, MiniMax language_boost)
LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nb": "Norwegian",
    "nl": "Dutch",
    "nn": "Nynorsk",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yue": "Chinese,Yue",
    "zh": "Chinese",
}

_KOKORO_VOICE = re.compile(r"^([a-z])[fm]_")


def primary_language(language: str) -> str:
    """Return the language subtag of a tag such as en-US."""
    return re.split(r"[-_]", language, maxsplit=1)[0].lower()


def voice_language(voice: str) -> str | None:
    """Return the language of a Kokoro-style voice ID."""
    match = _KOKORO_VOICE.match(voice)
    return KOKORO_VOICE_LANGUAGES.get(match.group(1)) if match else None


def unlisted_models(tts_models: Iterable[str], stt_models: Iterable[str]) -> list[str]:
    """Return the speech models missing from the language tables."""
    return sorted(
        {m for m in tts_models if m not in TTS_MODELS}
        | {m for m in stt_models if m not in STT_MODELS}
    )


def stt_languages(model_id: str) -> list[str]:
    """Return the languages a speech-to-text model understands."""
    return list(STT_MODELS.get(model_id, MULTILINGUAL))


def tts_languages(model_id: str, voices: tuple[str, ...]) -> list[str]:
    """Return the languages a text-to-speech model speaks with these voices."""
    info = TTS_MODELS.get(model_id)
    if info is None:
        return list(MULTILINGUAL)
    if info.per_voice:
        spoken = {voice_language(voice) for voice in voices}
        return [lang for lang in info.languages if lang in spoken]
    return list(info.languages)


def tts_voices(model_id: str, voices: tuple[str, ...], language: str) -> list[str]:
    """Return the voices of a text-to-speech model suitable for a language."""
    info = TTS_MODELS.get(model_id)
    if info is None or not info.per_voice:
        return list(voices)
    lang = primary_language(language)
    matching = [voice for voice in voices if voice_language(voice) == lang]
    return matching or list(voices)


def tts_language_hint(model_id: str, language: str) -> str | None:
    """Return the ``language`` value Venice expects for a TTS model, if any."""
    info = TTS_MODELS.get(model_id)
    lang = primary_language(language)
    if info is None or info.hint is None or lang not in info.languages:
        return None
    if info.hint is Hint.NAME:
        return LANGUAGE_NAMES.get(lang)
    # ISO 639-1 has a single code for Norwegian
    return "no" if lang == "nb" else lang
