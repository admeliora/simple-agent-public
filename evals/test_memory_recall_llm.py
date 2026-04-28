"""Real-LLM recall evals for cross-conversation memory.

These hit a live model and require provider credentials. Mark `slow` so the
default test pass stays cheap; run with:

    uv run pytest evals/test_memory_recall_llm.py -v -m slow
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from agent.core import make_agent
from agent.memory import MemoryCoordinator, MemoryStore, memory_scope_id
from agent.memory_mode import MemoryMode

load_dotenv()

pytestmark = pytest.mark.slow

MODEL = os.getenv("MEMORY_RECALL_MODEL", "anthropic:claude-haiku-4-5-20251001")

# A short conversation where the recall question depends on the FIRST user
# turn only — i.e. only modes that preserve early context can answer.
SEED_TURNS: list[tuple[str, str]] = [
    ("user", "My name is Alice and I live in Seattle."),
    ("assistant", "Nice to meet you, Alice in Seattle."),
    ("user", "I'm planning a trip to Tokyo in June 2026."),
    ("assistant", "Tokyo in June 2026, noted."),
    ("user", "I'd like sushi recommendations near Shibuya."),
    ("assistant", "Sukiyabashi Jiro Roppongi or Sushi Saito are popular."),
    ("user", "Also recommend a Hakone day trip itinerary."),
    ("assistant", "Train to Hakone-Yumoto, ropeway, then Lake Ashi cruise."),
]
RECALL_QUESTION = "What city do I live in?"


def _seed(coordinator: MemoryCoordinator, scope: str) -> None:
    for i in range(0, len(SEED_TURNS), 2):
        u_role, u_content = SEED_TURNS[i]
        a_role, a_content = SEED_TURNS[i + 1]
        coordinator.persist_exchange(
            scope,
            {"role": u_role, "content": u_content},
            {"role": a_role, "content": a_content},
        )


def _ask(mode: MemoryMode, tmp_path: Path) -> str:
    store = MemoryStore(str(tmp_path / f"recall-{mode.value}.db"))
    coord = MemoryCoordinator(
        mode=mode,
        store=store,
        summary_model_str=MODEL,
        summary_recent_window=2,  # force older facts to actually fold into summary
    )
    scope = memory_scope_id("alice", "session-1")
    _seed(coord, scope)
    agent = make_agent(model_str=MODEL, memory_mode=mode)
    prompt_messages = coord.build_prompt_messages(
        scope, [{"role": "user", "content": RECALL_QUESTION}]
    )
    result = agent.invoke({"messages": prompt_messages})
    return result["messages"][-1].content


def test_raw_mode_recalls_early_fact(tmp_path: Path):
    answer = _ask(MemoryMode.RAW, tmp_path)
    assert "Seattle" in answer


def test_summary_mode_recalls_early_fact(tmp_path: Path):
    answer = _ask(MemoryMode.SUMMARY, tmp_path)
    # The summary should preserve "Seattle" even though it fell out of the
    # 2-message recent window.
    assert "Seattle" in answer


def test_none_mode_does_not_recall_early_fact(tmp_path: Path):
    answer = _ask(MemoryMode.NONE, tmp_path)
    # With no memory, the model has no way to know — accept any non-Seattle
    # answer (refusal, hedge, hallucinated city). We only assert it can't have
    # leaked the seed.
    assert "Seattle" not in answer
