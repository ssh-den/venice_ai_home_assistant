# Privacy

The integration runs every request through the Venice AI cloud. This page
lists what leaves Home Assistant, what stays local and what Venice promises
about the data it receives.

## What is sent to Venice

| Feature | Sent to Venice |
| --- | --- |
| Conversation agent | The system prompt, the conversation so far (the last 50 messages unless **Conversation history** is lower), and, when **Control Home Assistant** is on, the names, areas and states of the entities exposed to Assist, the available tools and their results. |
| AI Task | The task instructions and the requested structure. |
| Speech-to-text | The recorded voice command as WAV audio and the pipeline language. |
| Text-to-speech | The text to speak, the voice, the speed and, for some models, the language. |
| `venice_ai.generate_image` | The image prompt and settings. |
| Background refresh | Every 12 hours: a check of the API key and requests for the model lists. |

Which entities the agent can see is controlled by Home Assistant under
**Settings → Voice assistants → Expose**. Entities that are not exposed are not
sent.

## What stays in Home Assistant

- The API key is stored in the config entry like any other integration
  credential. Diagnostics show only its last four characters and hide the
  system prompt.
- Home Assistant keeps the conversation history of a session; the integration
  stores no history of its own.
- Request, error and token counters live in memory and reset when the entry
  reloads.
- The integration does not log prompts, answers, transcriptions or tool
  arguments, only their sizes and tool names. Error messages returned by
  Venice are logged as they are.

## Privacy levels of Venice models

The model selectors show the privacy level Venice reports for each chat,
text-to-speech and speech-to-text model. Prefer **Private**, **TEE** or **E2EE**
models when the conversation or the voice commands are sensitive.

With **Private models only** on, the selectors list only Private, TEE and E2EE
models and refuse others. If Venice later lowers the privacy of a configured
model, Home Assistant shows a repair issue.

- **Anonymized**: Venice hides your identity from the model provider, but the
  provider processes the prompt under its own policies.
- **Private**: Venice or its zero-retention partners process the prompt and do
  not keep it. This rests on contracts.
- **TEE**: inference runs in an attested hardware enclave.
- **E2EE**: the client encrypts the prompt so that only the enclave can read
  it. The integration does not implement this yet, so E2EE-capable models are
  used like TEE models.

TEE and E2EE are available for text models only; speech models are either
private or anonymized.

According to its documentation, Venice does not store or log prompt and response
content for normal inference, but it does process metadata such as the API key
ID, model, token counts, timestamps and IP address for billing, rate limiting
and abuse prevention. Its terms state that user content is not used to train
models.

## Venice policies

- [Privacy policy](https://venice.ai/legal/privacy-policy)
- [Terms of service](https://venice.ai/legal/tos)
- [Privacy in Venice](https://venice.ai/privacy)
- [API privacy documentation](https://docs.venice.ai/overview/privacy)

These documents are the authoritative source; this page only summarizes them
and may lag behind changes.
