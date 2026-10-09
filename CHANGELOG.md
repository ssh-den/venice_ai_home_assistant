# Changelog

All notable changes to **Venice AI Conversation** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [2.0.0] — 2026-10-09

This release is published from the
[ssh-den/venice_ai_home_assistant](https://github.com/ssh-den/venice_ai_home_assistant)
fork and differs noticeably from upstream.

### Breaking changes
- Home Assistant 2025.9 or newer is required.
- The default chat model is now `e2ee-deepseek-v4-flash`; entries that have not
  saved a model in the options switch to it.
- The `todo` platform was removed.
- The Venice client is built on the OpenAI Python SDK, which replaces the
  hand-written HTTP client. Retry, timeout and HTTP pool constants were removed
  from `const.py`.

### Added
- Conversation answers are streamed to Assist as they are generated. Tool calls
  run straight from the stream, and model reasoning is stored as thinking
  content instead of being spoken.
- The chat model selector shows privacy (E2EE, TEE, private, anonymized), tool
  support and pricing reported by Venice.
- AI Task requests native JSON schema output from models that support it.
- `image_model` option and `model` field for `venice_ai.generate_image`; the
  model is validated against the models Venice offers.
- `request_timeout` and `stream_response` options, descriptions for every
  option and missing translations.
- Diagnostic sensors update as soon as a request finishes.
- Tests run against a real Home Assistant instance and cover every platform.
  black, ruff, mypy, pyright and pylint are configured in `pyproject.toml`.

### Changed
- The conversation entity relies on the Home Assistant chat log for history,
  the system prompt, `extra_system_prompt` and tool execution.
- Tools are not sent to models without function calling.
- Service actions are registered in `async_setup` and check that the entry is
  loaded.
- The API key is validated against an authenticated endpoint, since the model
  list is public.

### Fixed
- The `ai_task` action failed on every call.
- The user message was added to the conversation history twice, and the system
  prompt was sent twice.
- Streaming chat requests were not counted in the diagnostic sensors.
- `request_timeout` was ignored by most requests.
- Select options lost their values in the options flow.
- Downgraded config entries were accepted by the migration.

## [1.0.0] — upstream

### Added
- **DOC-1:** Module-level docstrings on all public client classes
  (`AsyncVeniceAIClient`, `ChatCompletions`, `Models`, `Speech`,
  `Transcriptions`, `Images`, `VeniceAIMetrics`).
- **DOC-2:** README **Operations** and **Troubleshooting** sections.
- **MAINT-3:** `FEATURE_MIN_VERSIONS` table in `const.py` referencing
  HA minimum versions for `ai_task`, `streaming_tts`,
  `conversation_entity`, and `sensor_total_increasing`.
- **MED-4:** Retry constants `MAX_RETRIES`, `RETRY_BASE_DELAY`,
  `RETRY_MAX_DELAY` extracted to `const.py`.
- **PERF-4:** `httpx.Limits` configured with `DEFAULT_HTTP_KEEPALIVE`
  and `DEFAULT_HTTP_MAX_CONNECTIONS` from `const.py`.
- **SEC-1:** API-key redaction helper (`_redact_api_key`) used by
  `diagnostics.py`, plus header-value sanitiser
  (`_sanitize_header_value`) in `client.py` to scrub CR/LF.
- **SEC-2:** Tool-call argument validation in `conversation.py`
  ensures tool args are JSON objects before invocation.
- **TEST-3:** `tests/test_schema.py` adds 16 schema-conversion tests
  covering `_format_venice_schema` and `_convert_schema_to_hashable`.
- **MAINT-2:** `async_migrate_entry` migrates older entries to the
  current `(version=1, minor_version=1)` schema.

### Changed
- **PERF-4:** Centralised HTTP timeout/keepalive defaults in
  `const.py` so they can be tuned without touching `client.py`.

### Fixed
- **MED-3:** Conversation streaming flag (`CONF_STREAM_RESPONSE`)
  honoured by the conversation entity; opt-in default off.
- **SEC-1 (regression):** `_sanitize_header_value` in `client.py`
  previously called `.strip()` and filtered every character with
  `ord(ch) >= 0x20` on the API key. That combination silently mutated
  previously-valid keys (e.g. by trimming surrounding whitespace or
  NBSP introduced during copy-paste) into byte-different strings, which
  Venice rejected with HTTP 401 on every request — observed as
  `Venice AI HTTP error 401:` in `client.py` and
  `Invalid API key (streaming chat)` in `conversation.py`. The helper
  now removes only `\r` and `\n` (the actual header-injection vectors)
  and the client stores the unmodified `api_key` on `self._api_key`,
  applying the scrub at header-construction time only. Users who
  re-entered their key during the affected window do not need to do so
  again. Reported against commit `64b115c`.

## [0.9.0] — Initial public release

### Added
- Conversation agent backed by Venice AI Chat Completions.
- AI Task entity (`ai_task.py`) using
  `genai.process_image`/`process_text`.
- TTS entity (`tts.py`) using Venice Speech API.
- STT entity (`stt.py`) using Venice Transcriptions API.
- Coordinator-driven sensor reporting token usage and last-error.
- Reauthentication and reconfiguration flows.
- Diagnostics export with redacted API key.
- 25 unit tests covering `client.py` and `venice_api.py`.

[2.0.0]: https://github.com/ssh-den/venice_ai_home_assistant
[1.0.0]: https://github.com/grasponcrypto/venice_ai
[0.9.0]: https://github.com/grasponcrypto/venice_ai/releases/tag/0.9.0
