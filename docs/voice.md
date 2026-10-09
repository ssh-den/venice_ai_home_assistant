# Voice assistant

Speech runs in the Venice cloud; nothing is downloaded to Home Assistant.

Venice does not report which languages a speech model supports, so the
integration keeps them in one table,
[`languages.py`](../custom_components/venice_ai/languages.py), compiled from each
vendor's documentation. A voice pipeline only offers engines that support its
language, and the language is passed to Venice in the form each model expects.

| Text-to-speech | Languages |
| --- | --- |
| `tts-kokoro` | English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese; voices follow the pipeline language |
| `tts-qwen3-0-6b`, `tts-qwen3-1-7b` | 10 languages, including Russian |
| `tts-xai-v1` | 16 languages, including Russian |
| `tts-inworld-1-5-max` | 15 languages, including Russian |
| `tts-elevenlabs-turbo-v2-5` | 32 languages, including Russian and Ukrainian |
| `tts-minimax-speech-02-hd` | 37 languages, including Russian and Ukrainian |
| `tts-gemini-3-1-flash` | 78 languages |
| `tts-gradium-v1` | English, French, German, Spanish, Portuguese |
| `tts-orpheus`, `tts-chatterbox-hd` | English |

| Speech-to-text | Languages |
| --- | --- |
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages, including Russian and Ukrainian |
| `openai/whisper-large-v3`, `fal-ai/wizper` | About 100 languages |
| `elevenlabs/scribe-v2` | About 100 languages |
| `stt-xai-v1` | 38 languages |

Models that Venice adds later are offered for every language until they are
added to the table; the log and the diagnostics list them.

Audio is requested as MP3, or WAV for models without MP3; Home Assistant
converts it for each speaker. Long answers are split at sentence boundaries,
so the first sentence plays while the rest is still being generated, and the
Venice limit of 4096 characters per request is never hit.
