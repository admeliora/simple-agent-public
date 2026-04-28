# Memory Harness

This harness demonstrates how different cross-conversation memory approaches change
what context is sent into the model for the same scripted conversation.

## Modes demonstrated

- `none`: only the current user turn is included
- `raw`: full persisted history is included
- `summary`: a running summary + the latest user turn are included

The harness is deterministic and **does not make external LLM calls**.

## Scripted conversation

The demo script includes a short conversation where the user shares:

- their name (`Alice`)
- a preference (`concise answers`)
- a travel destination (`Tokyo in June`)

The last turn asks for recall, and each memory mode shows a different context payload.

## Run

From repository root:

```bash
uv run memory-harness --mode all
```

Or run a single mode:

```bash
uv run memory-harness --mode raw
uv run memory-harness --mode summary
uv run memory-harness --mode none
```

## Testing checklist (copy/paste)

Use these commands from repo root to validate the harness and memory behavior:

```bash
# 1) run harness in all modes (deterministic output, no API calls)
uv run memory-harness --mode all

# 2) assert harness behavior via tests
uv run pytest evals/test_memory_modes.py -v

# 3) assert persisted memory behavior, including conversation isolation
uv run pytest evals/test_memory_persistence.py -v
```

What to verify:

- `none` output contains only `current_user_turn`.
- `raw` output shows the full scripted history.
- `summary` output shows `memory_summary` + `recent_user_turn`.
- Persistence tests confirm memory is reused across turns in the same conversation.
- User-scope tests confirm memory is shared across conversations for the same user.
- Isolation tests confirm memory from one user does **not** leak into another user.

## Interpreting output

- `none` is the baseline with no cross-conversation memory.
- `raw` maximizes fidelity but grows context size.
- `summary` is compact but can lose detail depending on summary quality.
