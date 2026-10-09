# Contributing

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/); the required
Python version is declared in `pyproject.toml`.

```bash
git clone https://github.com/ssh-den/venice_ai_home_assistant.git
cd venice_ai_home_assistant
uv sync
```

To try the integration in a development Home Assistant, copy
`custom_components/venice_ai` into its `config/custom_components` directory.

## Checks

Tests run against a real Home Assistant instance through
`pytest-homeassistant-custom-component`; the Venice client is mocked, so no API
key or network access is needed.

```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/black --check . && .venv/bin/mypy && .venv/bin/pyright && .venv/bin/pylint custom_components tests
```

All of them must pass before a change is merged. New behaviour needs tests.

## Style

- Type hints everywhere; the code must stay clean under mypy and pyright.
- Constants and defaults live in `const.py`.
- `strings.json` and `translations/en.json` are kept identical.
- Comments only where the code cannot speak for itself.
- Never log the API key.

## Speech model languages

Venice does not report the languages of its speech models, so
`custom_components/venice_ai/languages.py` lists them. When Venice adds or
changes a TTS or STT model, which the integration logs and shows in the
diagnostics, add its languages from the vendor's documentation, note the source
next to the entry and say whether the model takes the language as a code or a
name. Update the tables in `docs/voice.md` to match.

## Releases

1. Describe the changes in `CHANGELOG.md`.
2. Bump the version in `manifest.json` and `pyproject.toml`.
3. Tag the release with the same version.

## Reporting bugs

Open an issue at
[ssh-den/venice_ai_home_assistant](https://github.com/ssh-den/venice_ai_home_assistant/issues)
with the Home Assistant and integration versions, a redacted debug log and steps
to reproduce.

## License

Contributions are licensed under the terms in [LICENSE](LICENSE).
