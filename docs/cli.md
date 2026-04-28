# CLI guide

Run the agent as an interactive terminal chat. See the [core README](../README.md) for initial setup.

## Start

```bash
# Default model (Anthropic Claude Haiku)
uv run chat

# OpenAI
uv run chat --model openai:gpt-4o

# Google
uv run chat --model google_genai:gemini-2.5-flash

# Custom system prompt
uv run chat --system "You are a helpful coding assistant."

# Memory mode scaffold
uv run chat --memory-mode raw --conversation-id alice
uv run chat --memory-mode summary --conversation-id alice
```

You can set `MEMORY_DB_PATH` to customize SQLite storage location:

```bash
MEMORY_DB_PATH=.data/memory.db uv run chat --memory-mode raw --conversation-id alice
```

Type `quit` or `exit` to end the session.

## How it works

`src/agent/cli.py` keeps a running in-memory `messages` list when `--memory-mode none`. In `raw` and `summary` modes it reads/writes persisted conversation memory keyed by `--conversation-id` and builds prompt messages from stored state.

## Relevant files

```
src/agent/
├── core.py     # agent factory (shared)
└── cli.py      # REPL loop, argument parsing
```
