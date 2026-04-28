from __future__ import annotations

import argparse
from dataclasses import dataclass

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory mode comparison harness")
    parser.add_argument(
        "--mode",
        default="all",
        help="none, raw, summary, or all",
    )
    args = parser.parse_args()

    if args.mode == "all":
        outputs = [run(mode) for mode in MemoryMode]
        print("\n\n".join(outputs))
        return

    mode = parse_memory_mode(args.mode)
    print(run(mode))


if __name__ == "__main__":
    main()
