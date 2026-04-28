# Memory Harness

This harness demonstrates how different cross-conversation memory approaches
change what context is sent into the model for the same scripted conversation.
It drives the **real** `MemoryCoordinator` (not a parallel simulation) so the
output reflects exactly what the CLI/server would build at runtime.

## Modes demonstrated

- `none`: only the current user turn is included
- `raw`: full persisted history is included
- `summary`: a rolling summary of older turns + the most recent `K` raw turns
  (no overlap — once a turn falls out of the recent window it is folded into
  the summary and dropped from the raw tail)

The harness is deterministic and **does not make external LLM calls** — it uses
a stand-in summary updater that extracts a few well-known fact patterns from
the scripted conversation.

## Demos

```bash
# side-by-side — runs all three modes on the same scripted convo and prints
# per-mode prompt contents + char count breakdown (summary / recent / raw)
uv run memory-harness --demo diff

# prompt-size growth across many turns — shows raw growing ~linearly while
# summary stays bounded by recent_window + summary length
uv run memory-harness --demo growth --max-turns 30

# user-scoped sharing + cross-user isolation
uv run memory-harness --demo user-scope

# single mode (use --mode none|raw|summary)
uv run memory-harness --demo context --mode summary

# all demos back-to-back
uv run memory-harness --demo all
```

The `--recent-window` flag controls the summary mode's tail size (default 4).

## Scripted conversation

15 turns, 8 user / 7 assistant. The user shares:

- name (`Alice`)
- preference (`concise answers`)
- role (`software engineer, payments`)
- destination (`Tokyo, June 2026`)
- a few interest signals (sushi near Shibuya, weather, day trips)

The 15th turn is a recall question: "what is my name and where am I traveling,
and when?" — designed so that only `raw` and `summary` can answer fully.

## Testing checklist

```bash
# 1) deterministic harness output (no API calls)
uv run memory-harness --demo all

# 2) unit + persistence tests (default, fast)
uv run pytest evals/ -v

# 3) real-LLM recall tests (require credentials)
uv run pytest evals/ -m slow -v
```

What to verify in the deterministic output:

- `none`: prompt is exactly the recall turn (1 message).
- `raw`: prompt contains every prior persisted turn (15 messages here).
- `summary`: prompt contains a `system` summary block + only the last
  `recent_window` raw messages + the recall turn — and the oldest messages
  appear **only** inside the summary block (no overlap with the raw tail).
- `--demo growth` shows `raw_chars` rising roughly linearly with turn count
  while `summary_chars` stays roughly flat.
- `--demo user-scope` reports `True` for both same-user sharing and
  cross-user isolation in all four assertions.

## Troubleshooting

If `memory-harness` fails with `ModuleNotFoundError: No module named 'agent'`:

1. Reinstall project + scripts into the virtualenv:

   ```bash
   uv sync --reinstall-package take-home
   ```

2. Run via uv (recommended):

   ```bash
   uv run memory-harness --demo diff
   ```

3. Or run module form directly:

   ```bash
   PYTHONPATH=src .venv/bin/python -m agent.memory_harness --demo diff
   ```

## Interpreting output

- `none` is the baseline with no cross-conversation memory.
- `raw` maximizes fidelity but grows context size linearly — fine for short
  sessions, expensive at scale.
- `summary` keeps total prompt size bounded but quality depends on the summary
  updater (the harness uses a deterministic stub; in production this is an
  LLM call). Older turns survive only via the summary text — verify the
  summary preserves the durable facts you need.
