# Configuration

One Venice AI entry holds the API key and a set of services. Each service has
its own device and entity; add as many as you need, for example a second
conversation agent with a cheaper model or another prompt.

## Entry settings

**Configure** on the entry.

| Setting | Default | Description |
| --- | --- | --- |
| Private models only | off | List only Private and TEE models in the service settings. A repair issue reports services that use other models. |
| Image model | Venice default | Model used by `venice_ai.generate_image` when no model is given. |
| Request timeout | 120 s | Timeout for every Venice AI request; for streamed answers, the wait between chunks. |

## Conversation agent

The first step holds the name, the instructions, the Home Assistant APIs and the
model. With **Recommended settings** on, the advanced step is skipped and its
defaults apply.

| Setting | Default | Description |
| --- | --- | --- |
| Instructions | a short assistant prompt | Sent with every conversation. Supports templates. |
| Control Home Assistant | Assist | LLM APIs the agent may use. Home Assistant adds the exposed entities and their states to the prompt. |
| Model | `e2ee-deepseek-v4-flash` | Venice text model. Pick one tagged `tools` to control devices. |
| Max tokens | 512 | Length limit of an answer. |
| Temperature / Top P | 1.0 / 1.0 | Sampling. |
| Disable thinking | on | Ask reasoning models to skip reasoning. |
| Thinking tags | `think` | Names of the tags that wrap reasoning in answers, separated by commas. |
| Strip thinking | on | Keep the reasoning in those tags out of the spoken answer. |
| Venice system prompt | off | Let Venice add its own system prompt next to the instructions. |
| Stream responses | on | Stream answers to Assist while they are generated. |
| Max tool iterations | 5 | Model calls allowed per turn while using tools. |
| Conversation history | 50 | Earlier messages sent with each request; the current question is always sent. |

Reasoning that Venice returns separately from the answer is kept as thinking
content and never spoken.

## AI Task

The same model, sampling and reasoning settings as the conversation agent, plus:

| Setting | Default | Description |
| --- | --- | --- |
| Structured output instructions | asks for JSON only | Sent with the JSON schema of the requested structure to models without native structured output. Models with it get the schema directly. |

## Text-to-speech

The first step picks the model, labeled with its privacy level, the number of
languages it speaks and its price per million characters; the second step picks one of its voices and the speed.
Kokoro voices show their language. [voice.md](voice.md) lists the languages of
every model.

## Speech-to-text

One setting: the model, labeled with its privacy level and its price per minute
of audio.

## Model lists

The service settings list the models Venice offers at that moment. The
integration also refreshes the lists every 12 hours and raises a repair issue
when a configured model disappears.
