from pathlib import Path

from agent.memory import MemoryCoordinator, MemoryStore
from agent.memory_mode import MemoryMode


def fake_summary_updater(existing_summary: str, new_messages: list[dict[str, str]], model_str: str) -> str:
    _ = model_str
    latest_user = next(m["content"] for m in new_messages if m["role"] == "user")
    if existing_summary:
        return f"{existing_summary} | {latest_user}"
    return latest_user


def test_raw_mode_reuses_persisted_history(tmp_path: Path):
    store = MemoryStore(str(tmp_path / "raw.db"))
    memory = MemoryCoordinator(
        mode=MemoryMode.RAW,
        store=store,
        summary_model_str="openai:gpt-4o-mini",
        summary_updater=fake_summary_updater,
    )

    conversation_id = "user-1"
    memory.persist_exchange(
        conversation_id,
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you."},
    )

    prompt = memory.build_prompt_messages(
        conversation_id,
        [{"role": "user", "content": "What is my name?"}],
    )

    assert prompt == [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]


def test_summary_mode_adds_summary_and_recent_turns(tmp_path: Path):
    store = MemoryStore(str(tmp_path / "summary.db"))
    memory = MemoryCoordinator(
        mode=MemoryMode.SUMMARY,
        store=store,
        summary_model_str="openai:gpt-4o-mini",
        summary_updater=fake_summary_updater,
        summary_recent_window=2,
    )

    conversation_id = "user-2"
    memory.persist_exchange(
        conversation_id,
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "Got it."},
    )

    prompt = memory.build_prompt_messages(
        conversation_id,
        [{"role": "user", "content": "Plan a Tokyo trip"}],
    )

    assert prompt[0]["role"] == "system"
    assert "I prefer concise answers." in prompt[0]["content"]
    assert prompt[1:] == [
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "Plan a Tokyo trip"},
    ]


def test_none_mode_passes_through_messages(tmp_path: Path):
    store = MemoryStore(str(tmp_path / "none.db"))
    memory = MemoryCoordinator(
        mode=MemoryMode.NONE,
        store=store,
        summary_model_str="openai:gpt-4o-mini",
        summary_updater=fake_summary_updater,
    )

    incoming = [{"role": "user", "content": "hello"}]
    assert memory.build_prompt_messages("conv", incoming) == incoming


def test_memory_is_isolated_across_conversations(tmp_path: Path):
    store = MemoryStore(str(tmp_path / "isolation.db"))
    memory = MemoryCoordinator(
        mode=MemoryMode.RAW,
        store=store,
        summary_model_str="openai:gpt-4o-mini",
        summary_updater=fake_summary_updater,
    )

    memory.persist_exchange(
        "conversation-a",
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Noted: blue."},
    )
    memory.persist_exchange(
        "conversation-b",
        {"role": "user", "content": "My favorite color is green."},
        {"role": "assistant", "content": "Noted: green."},
    )

    prompt_a = memory.build_prompt_messages(
        "conversation-a",
        [{"role": "user", "content": "What is my favorite color?"}],
    )
    prompt_b = memory.build_prompt_messages(
        "conversation-b",
        [{"role": "user", "content": "What is my favorite color?"}],
    )

    assert any("blue" in msg["content"] for msg in prompt_a)
    assert not any("green" in msg["content"] for msg in prompt_a)
    assert any("green" in msg["content"] for msg in prompt_b)
    assert not any("blue" in msg["content"] for msg in prompt_b)
