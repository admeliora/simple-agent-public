# simple-agent

A minimal LLM agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). Supports OpenAI, Anthropic, and Google models out of the box. Two ways to run it — pick one:

- [CLI guide](docs/cli.md) — interactive terminal chat
- [Fullstack guide](docs/fullstack.md) — FastAPI server + React frontend

### Choosing between the two approaches

- **CLI (`uv run chat`)**
  - Best for local experimentation, prompt iteration, and debugging.
  - Minimal setup and fastest feedback loop.
  - Tradeoff: single-terminal UX and no built-in multi-user HTTP interface.

- **Fullstack (`uv run serve` + React frontend)**
  - Best when you want a browser UI and API surface for app integration.
  - Supports client/server separation and easier extension for product use-cases.
  - Tradeoff: more moving parts (backend + frontend processes) and operational complexity.

---

## Core agent

The agent lives in `src/agent/core.py` and exposes a single factory:

```python
from agent.core import make_agent

agent = make_agent(
    model_str="anthropic:claude-haiku-4-5-20251001",  # provider:model
    system_prompt=None,                                # optional override
)
```

It wraps LangChain's `init_chat_model` + `create_deep_agent` and returns a compiled LangGraph agent that supports `.invoke()`, `.stream()`, and `.astream()`.

## Supported providers

| Provider  | Model string example                            | Required env var    |
|-----------|-------------------------------------------------|---------------------|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` (default) | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai:gpt-4o`                                 | `OPENAI_API_KEY`    |
| Google    | `google_genai:gemini-2.5-flash`                 | `GOOGLE_API_KEY`    |

Any model supported by LangChain's [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) works — just pass the `provider:model` string.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM provider API key

## Initial setup

```bash
git clone https://github.com/valkai-tech/simple-agent-public.git
cd simple-agent-public
uv sync
cp .env.example .env
# Fill in your API key(s) in .env
```


## Memory mode toggle

Cross-conversation memory modes:

- `none` (default)
- `raw`
- `summary`

CLI:

```bash
uv run chat --memory-mode raw
```

Server:

```bash
MEMORY_MODE=summary uv run serve
```

Use `MEMORY_DB_PATH` to control where conversation memory is stored (SQLite):

```bash
MEMORY_DB_PATH=.data/memory.db MEMORY_MODE=raw uv run serve
```

Memory scope rules:

- Provide a user ID (`--user-id` in CLI or `user_id` in server requests) to share memory across that user's conversations.
- If no user ID is provided, memory falls back to conversation-level scope (`conversation_id`).

## Memory harness

To compare memory modes with a deterministic scripted conversation:

```bash
uv run memory-harness --mode all
```

To demonstrate same-user sharing and cross-user isolation:

```bash
uv run memory-harness --demo user-scope
```

See `harness/README.md` for details.

The harness is deterministic and demonstrates how `none`, `raw`, and `summary` produce different prompt contexts for the same scripted conversation.

## Running evals

```bash
uv run pytest evals/ -v
```

Evals make real LLM calls (not mocked) to verify provider integration end-to-end.

## Project structure

```
simple-agent/
├── README.md               # this file — core concepts
├── docs/
│   ├── cli.md              # CLI usage guide
│   └── fullstack.md        # server + frontend guide
├── pyproject.toml          # uv project config and dependencies
├── .env.example            # API key template
├── src/
│   └── agent/
│       ├── core.py         # agent factory (shared by both approaches)
│       ├── cli.py          # CLI entry point
│       └── server.py       # FastAPI server entry point
├── frontend/               # React chat UI
└── evals/
    └── test_agent.py       # pytest evals
```
