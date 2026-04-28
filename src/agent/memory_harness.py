from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass

from agent.memory import MemoryCoordinator, MemoryStore, memory_scope_id
from agent.memory_mode import MemoryMode, parse_memory_mode


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


SCRIPTED_TURNS = [
    Turn("user", "My name is Alice and I prefer concise answers."),
    Turn("assistant", "Nice to meet you, Alice. I will keep answers concise."),
    Turn("user", "I am planning a Tokyo trip in June."),
    Turn("assistant", "Great, I can help you plan your Tokyo trip."),
    Turn("user", "Remind me later: what is my name and where am I traveling?"),
]


def make_summary(turns: list[Turn]) -> str:
    """Very small deterministic summary for docs/testing harness.

    This keeps the harness free of external LLM calls.
    """

    facts: list[str] = []
    for turn in turns:
        text = turn.content.lower()
        if turn.role == "user" and "my name is" in text:
            facts.append("User name: Alice")
        if turn.role == "user" and "tokyo" in text:
            facts.append("Travel destination: Tokyo in June")
        if turn.role == "user" and "prefer concise" in text:
            facts.append("Preference: concise answers")

    if not facts:
        return "No durable facts captured yet."
    return " | ".join(dict.fromkeys(facts))


def build_context(mode: MemoryMode, turns: list[Turn]) -> list[str]:
    if mode == MemoryMode.NONE:
        return [f"current_user_turn: {turns[-1].content}"]

    if mode == MemoryMode.RAW:
        return [f"{t.role}: {t.content}" for t in turns]

    summary = make_summary(turns[:-1])
    return [
        f"memory_summary: {summary}",
        f"recent_user_turn: {turns[-1].content}",
    ]


def run(mode: MemoryMode) -> str:
    lines = [f"=== Memory mode: {mode.value} ==="]
    lines.extend(build_context(mode, SCRIPTED_TURNS))
    return "\n".join(lines)


def _harness_summary_updater(existing_summary: str, new_messages: list[dict[str, str]], model_str: str) -> str:
    _ = model_str
    latest_user = next(m["content"] for m in new_messages if m["role"] == "user")
    if existing_summary:
        return f"{existing_summary} | {latest_user}"
    return latest_user


def run_user_scope_demo() -> str:
    """Demonstrate user-scoped memory sharing + cross-user isolation deterministically."""
    user_a_seed = [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Noted: blue."},
    ]
    user_b_seed = [
        {"role": "user", "content": "My favorite color is green."},
        {"role": "assistant", "content": "Noted: green."},
    ]
    follow_up_user_turn = {"role": "user", "content": "What is my favorite color?"}

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(f"{tmpdir}/harness.db")
        memory = MemoryCoordinator(
            mode=MemoryMode.RAW,
            store=store,
            summary_model_str="openai:gpt-4o-mini",
            summary_updater=_harness_summary_updater,
        )

        memory.persist_exchange(
            memory_scope_id("user-a", "conversation-1"),
            user_a_seed[0],
            user_a_seed[1],
        )
        memory.persist_exchange(
            memory_scope_id("user-b", "conversation-1"),
            user_b_seed[0],
            user_b_seed[1],
        )

        prompt_a = memory.build_prompt_messages(
            memory_scope_id("user-a", "conversation-2"),
            [follow_up_user_turn],
        )
        prompt_b = memory.build_prompt_messages(
            memory_scope_id("user-b", "conversation-2"),
            [follow_up_user_turn],
        )

    shared_for_user_a = any("blue" in msg["content"] for msg in prompt_a)
    isolated_for_user_a = not any("green" in msg["content"] for msg in prompt_a)
    shared_for_user_b = any("green" in msg["content"] for msg in prompt_b)
    isolated_for_user_b = not any("blue" in msg["content"] for msg in prompt_b)

    lines = [
        "=== User scope demo ===",
        "Seed conversations (persisted):",
        "conversation-1 / user-a",
        "  user: My favorite color is blue.",
        "  assistant: Noted: blue.",
        "conversation-1 / user-b",
        "  user: My favorite color is green.",
        "  assistant: Noted: green.",
        "",
        "Follow-up conversations (new conversation_id=conversation-2):",
        "conversation-2 / user-a prompt built from persistence:",
        *[f"  {msg['role']}: {msg['content']}" for msg in prompt_a],
        "conversation-2 / user-b prompt built from persistence:",
        *[f"  {msg['role']}: {msg['content']}" for msg in prompt_b],
        "",
        f"shared_across_conversations_for_user_a: {shared_for_user_a}",
        f"isolated_from_user_b_for_user_a: {isolated_for_user_a}",
        f"shared_across_conversations_for_user_b: {shared_for_user_b}",
        f"isolated_from_user_a_for_user_b: {isolated_for_user_b}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory mode comparison harness")
    parser.add_argument(
        "--demo",
        choices=["context", "user-scope", "all"],
        default="context",
        help="context = compare none/raw/summary context payloads; user-scope = verify per-user sharing/isolation",
    )
    parser.add_argument(
        "--mode",
        default="all",
        help="none, raw, summary, or all",
    )
    args = parser.parse_args()

    if args.demo == "user-scope":
        print(run_user_scope_demo())
        return

    if args.demo == "all":
        outputs = [run(mode) for mode in MemoryMode]
        outputs.append(run_user_scope_demo())
        print("\n\n".join(outputs))
        return

    if args.mode == "all":
        outputs = [run(mode) for mode in MemoryMode]
        print("\n\n".join(outputs))
        return

    mode = parse_memory_mode(args.mode)
    print(run(mode))


if __name__ == "__main__":
    main()
