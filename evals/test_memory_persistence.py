from pathlib import Path

import pytest

from agent.memory import MemoryCoordinator, MemoryStore, memory_scope_id
from agent.memory_mode import MemoryMode


def fake_summary_updater(existing_summary: str, new_messages: list[dict[str, str]], model_str: str) -> str:
    """Append-only stand-in: concatenates new user content onto the running summary."""
    _ = model_str
    user_bits = [m["content"] for m in new_messages if m["role"] == "user"]
    delta = " | ".join(user_bits)
    if existing_summary and delta:
        return f"{existing_summary} | {delta}"
    return existing_summary or delta


def newest_wins_summary_updater(existing_summary: str, new_messages: list[dict[str, str]], model_str: str) -> str:
    """Replaces summary with the latest user content — tests contradiction handling."""
    _ = model_str
    _ = existing_summary
    user_bits = [m["content"] for m in new_messages if m["role"] == "user"]
    return user_bits[-1] if user_bits else existing_summary


def bounded_summary_updater(existing_summary: str, new_messages: list[dict[str, str]], model_str: str) -> str:
    """Simulates an LLM compressing the summary down to a fixed length budget.

    Real LLMs produce roughly bounded-length summaries; the fake append-only
    updater used elsewhere in this file does not, so this stand-in is needed
    to test the architectural boundedness property.
    """
    _ = model_str
    _ = new_messages
    return (existing_summary or "facts: …")[-200:]


def _make(mode: MemoryMode, tmp_path: Path, name: str, **kwargs) -> MemoryCoordinator:
    store = MemoryStore(str(tmp_path / f"{name}.db"))
    return MemoryCoordinator(
        mode=mode,
        store=store,
        summary_model_str="harness:deterministic",
        summary_updater=kwargs.pop("summary_updater", fake_summary_updater),
        **kwargs,
    )


def test_raw_mode_reuses_persisted_history(tmp_path: Path):
    memory = _make(MemoryMode.RAW, tmp_path, "raw")
    scope = "user-1"
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you."},
    )
    prompt = memory.build_prompt_messages(
        scope, [{"role": "user", "content": "What is my name?"}]
    )
    assert prompt == [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]


def test_summary_mode_does_not_summarize_until_window_overflows(tmp_path: Path):
    memory = _make(MemoryMode.SUMMARY, tmp_path, "small", summary_recent_window=4)
    scope = "user-2"
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "Got it."},
    )

    prompt = memory.build_prompt_messages(
        scope, [{"role": "user", "content": "Plan a Tokyo trip"}]
    )
    # Only 2 messages persisted, window=4 → nothing folded yet, no summary block.
    assert all(m["role"] != "system" for m in prompt)
    assert prompt == [
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "Plan a Tokyo trip"},
    ]


def test_summary_mode_folds_only_messages_outside_recent_window(tmp_path: Path):
    memory = _make(MemoryMode.SUMMARY, tmp_path, "fold", summary_recent_window=2)
    scope = "user-3"
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "Got it."},
    )
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "I'm planning a Tokyo trip."},
        {"role": "assistant", "content": "Tokyo, noted."},
    )

    prompt = memory.build_prompt_messages(
        scope, [{"role": "user", "content": "What's my preference?"}]
    )

    # Summary block contains the older messages...
    assert prompt[0]["role"] == "system"
    assert "I prefer concise answers." in prompt[0]["content"]
    # ...and the raw tail is exactly the most-recent window, no overlap.
    assert prompt[1:] == [
        {"role": "user", "content": "I'm planning a Tokyo trip."},
        {"role": "assistant", "content": "Tokyo, noted."},
        {"role": "user", "content": "What's my preference?"},
    ]
    # No-overlap invariant: oldest user msg must NOT appear as a non-system message.
    assert all(
        not (m["role"] == "user" and m["content"] == "I prefer concise answers.")
        for m in prompt[1:]
    )


def test_summary_persists_messages_older_than_window_via_summary_only(tmp_path: Path):
    memory = _make(MemoryMode.SUMMARY, tmp_path, "older", summary_recent_window=2)
    scope = "user-4"
    # 3 paired exchanges → 6 messages → window=2 means 4 should fold into summary.
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hi Alice."},
    )
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "I'm a software engineer."},
        {"role": "assistant", "content": "Got it."},
    )
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "Tokyo trip in June."},
        {"role": "assistant", "content": "Noted."},
    )

    prompt = memory.build_prompt_messages(
        scope, [{"role": "user", "content": "Recall my name."}]
    )

    # Earliest message survives ONLY via the summary block.
    summary_block = next(m for m in prompt if m["role"] == "system")
    assert "My name is Alice." in summary_block["content"]
    # …and is absent from the raw tail.
    raw_tail = [m for m in prompt if m["role"] != "system"]
    assert all("My name is Alice." not in m["content"] for m in raw_tail)


def test_summary_handles_contradictions_using_newest_wins(tmp_path: Path):
    memory = _make(
        MemoryMode.SUMMARY,
        tmp_path,
        "contradict",
        summary_recent_window=2,
        summary_updater=newest_wins_summary_updater,
    )
    scope = "user-5"
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "OK."},
    )
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "Actually I prefer detailed answers."},
        {"role": "assistant", "content": "Switched to detailed."},
    )
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "Tell me about Tokyo."},
        {"role": "assistant", "content": "Sure."},
    )

    prompt = memory.build_prompt_messages(
        scope, [{"role": "user", "content": "What's my preference?"}]
    )
    summary_block = next(m for m in prompt if m["role"] == "system")
    assert "detailed answers" in summary_block["content"]
    assert "concise answers" not in summary_block["content"]


def test_summary_updater_failure_does_not_lose_persisted_messages(tmp_path: Path):
    def boom(*_args, **_kwargs):
        raise RuntimeError("LLM unavailable")

    store = MemoryStore(str(tmp_path / "boom.db"))
    memory = MemoryCoordinator(
        mode=MemoryMode.SUMMARY,
        store=store,
        summary_model_str="harness:deterministic",
        summary_updater=boom,
        summary_recent_window=2,
    )
    scope = "user-6"
    # Two exchanges → 4 messages, with window=2 the older 2 should attempt to fold.
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "First message."},
        {"role": "assistant", "content": "OK1."},
    )
    memory.persist_exchange(
        scope,
        {"role": "user", "content": "Second message."},
        {"role": "assistant", "content": "OK2."},
    )

    # Messages must still be persisted even though the summary updater raised.
    assert store.message_count(scope) == 4
    # No summary written.
    assert store.get_summary(scope) == ""


def test_none_mode_persists_nothing(tmp_path: Path):
    store = MemoryStore(str(tmp_path / "none.db"))
    memory = MemoryCoordinator(
        mode=MemoryMode.NONE,
        store=store,
        summary_model_str="harness:deterministic",
        summary_updater=fake_summary_updater,
    )
    memory.persist_exchange(
        "scope-x",
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    )
    assert store.message_count("scope-x") == 0
    assert store.get_summary("scope-x") == ""


def test_none_mode_passes_through_messages(tmp_path: Path):
    memory = _make(MemoryMode.NONE, tmp_path, "pass")
    incoming = [{"role": "user", "content": "hello"}]
    assert memory.build_prompt_messages("conv", incoming) == incoming


def test_raw_grows_summary_stays_bounded(tmp_path: Path):
    """With a compressing updater (production proxy), summary stays bounded
    while raw grows linearly. Confirms the architectural invariant."""
    raw = _make(MemoryMode.RAW, tmp_path, "raw-grow")
    summ = _make(
        MemoryMode.SUMMARY,
        tmp_path,
        "summ-grow",
        summary_recent_window=4,
        summary_updater=bounded_summary_updater,
    )
    scope_raw = "scope-raw"
    scope_summ = "scope-summ"

    body = "x" * 200
    snapshots: list[tuple[int, int]] = []
    for i in range(40):
        u = {"role": "user", "content": f"u{i}: {body}"}
        a = {"role": "assistant", "content": f"a{i}: {body}"}
        raw.persist_exchange(scope_raw, u, a)
        summ.persist_exchange(scope_summ, u, a)
        if (i + 1) in (5, 20, 40):
            recall = [{"role": "user", "content": "recall"}]
            snapshots.append(
                (
                    raw.build_prompt(scope_raw, recall).total_chars,
                    summ.build_prompt(scope_summ, recall).total_chars,
                )
            )

    raw_5, summ_5 = snapshots[0]
    raw_40, summ_40 = snapshots[2]

    # Raw grows roughly linearly with turns (8x more turns ⇒ ~5x+ more chars).
    assert raw_40 > raw_5 * 5
    # Summary stays roughly flat (bounded by tail + bounded updater output).
    assert summ_40 < summ_5 * 2
    # And raw is decisively larger than summary at scale.
    assert raw_40 > summ_40 * 5

    # Architectural invariant: summary's raw tail is at most window+1 messages.
    summ_prompt = summ.build_prompt(scope_summ, [{"role": "user", "content": "recall"}])
    summ_tail = [m for m in summ_prompt.messages if m["role"] != "system"]
    assert len(summ_tail) <= 4 + 1


def test_memory_is_shared_across_conversations_for_same_user(tmp_path: Path):
    memory = _make(MemoryMode.RAW, tmp_path, "isolation")
    shared_scope = memory_scope_id("user-42", "conversation-a")
    memory.persist_exchange(
        shared_scope,
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Noted: blue."},
    )

    prompt_a = memory.build_prompt_messages(
        memory_scope_id("user-42", "conversation-a"),
        [{"role": "user", "content": "What is my favorite color?"}],
    )
    prompt_b = memory.build_prompt_messages(
        memory_scope_id("user-42", "conversation-b"),
        [{"role": "user", "content": "What is my favorite color?"}],
    )

    assert any("blue" in msg["content"] for msg in prompt_a)
    assert any("blue" in msg["content"] for msg in prompt_b)


def test_memory_is_isolated_between_different_users(tmp_path: Path):
    memory = _make(MemoryMode.RAW, tmp_path, "isolation-users")
    memory.persist_exchange(
        memory_scope_id("user-a", "conversation-1"),
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Noted: blue."},
    )
    memory.persist_exchange(
        memory_scope_id("user-b", "conversation-99"),
        {"role": "user", "content": "My favorite color is green."},
        {"role": "assistant", "content": "Noted: green."},
    )

    prompt_a = memory.build_prompt_messages(
        memory_scope_id("user-a", "conversation-2"),
        [{"role": "user", "content": "What is my favorite color?"}],
    )
    prompt_b = memory.build_prompt_messages(
        memory_scope_id("user-b", "conversation-100"),
        [{"role": "user", "content": "What is my favorite color?"}],
    )

    assert any("blue" in msg["content"] for msg in prompt_a)
    assert not any("green" in msg["content"] for msg in prompt_a)
    assert any("green" in msg["content"] for msg in prompt_b)
    assert not any("blue" in msg["content"] for msg in prompt_b)
