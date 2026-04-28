from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent.memory import MemoryCoordinator, MemoryStore, memory_scope_id
from agent.memory_mode import MemoryMode, parse_memory_mode


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


# 15 turns: 8 user / 7 assistant. Pairs (1,2)…(13,14) get persisted as
# user+assistant exchanges; turn 15 is the recall question used to build
# the next-call prompt under each mode.
SCRIPTED_TURNS: list[Turn] = [
    Turn("user", "Hi! My name is Alice and I prefer concise answers."),
    Turn("assistant", "Nice to meet you, Alice. I'll keep things brief."),
    Turn("user", "I'm a software engineer working on payments infrastructure."),
    Turn("assistant", "Got it — payments infra."),
    Turn("user", "I'm planning a Tokyo trip in June 2026."),
    Turn("assistant", "Tokyo in June 2026, noted."),
    Turn("user", "I'd love sushi recommendations near Shibuya."),
    Turn("assistant", "Sukiyabashi Jiro Roppongi or Sushi Saito are popular picks."),
    Turn("user", "What's the weather like in Tokyo in June?"),
    Turn("assistant", "Warm and humid, often rainy — bring a light jacket and umbrella."),
    Turn("user", "Any cherry blossom advice for that time?"),
    Turn("assistant", "Cherry blossoms peak in late March / early April — June is too late."),
    Turn("user", "Got it. Easy day trip from Tokyo?"),
    Turn("assistant", "The Hakone region near Mt Fuji is the easiest, accessible by train."),
    Turn("user", "Remind me later: what is my name and where am I traveling, and when?"),
]


def _harness_summary_updater(
    existing_summary: str,
    new_messages: list[dict[str, str]],
    model_str: str,
) -> str:
    """Deterministic stand-in for an LLM summary updater.

    Extracts a few well-known fact patterns from the new (out-of-window) turns
    and merges them with the prior summary. Newest occurrences win on conflict.
    """
    _ = model_str
    facts: dict[str, str] = {}
    if existing_summary:
        for line in existing_summary.split(" | "):
            if "=" in line:
                k, v = line.split("=", 1)
                facts[k.strip()] = v.strip()

    for m in new_messages:
        if m["role"] != "user":
            continue
        text = m["content"].lower()
        if "my name is" in text:
            facts["user_name"] = "Alice"
        if "prefer concise" in text:
            facts["preference"] = "concise answers"
        if "software engineer" in text and "payments" in text:
            facts["role"] = "software engineer (payments)"
        if "tokyo" in text and ("trip" in text or "traveling" in text):
            facts["destination"] = "Tokyo (June 2026)"
        if "sushi" in text and "shibuya" in text:
            facts["interest"] = "sushi near Shibuya"

    if not facts:
        return existing_summary
    return " | ".join(f"{k}={v}" for k, v in facts.items())


def _make_coordinator(mode: MemoryMode, db_path: Path, recent_window: int) -> MemoryCoordinator:
    store = MemoryStore(str(db_path))
    return MemoryCoordinator(
        mode=mode,
        store=store,
        summary_model_str="harness:deterministic",
        summary_updater=_harness_summary_updater,
        summary_recent_window=recent_window,
    )


def _replay_pairs_then_recall(
    coordinator: MemoryCoordinator,
    scope_id: str,
    turns: list[Turn],
):
    """Persist all paired user+assistant turns, then build a prompt for the
    final user turn (which must be a 'lone' user turn at the tail).
    """
    if turns[-1].role != "user":
        raise ValueError("scripted convo must end on a user turn for the recall prompt")

    pairs = turns[:-1]
    if len(pairs) % 2 != 0:
        raise ValueError("paired turns (all but the last user turn) must be even-length")

    for i in range(0, len(pairs), 2):
        u, a = pairs[i], pairs[i + 1]
        if u.role != "user" or a.role != "assistant":
            raise ValueError(f"expected user/assistant pair at index {i}")
        coordinator.persist_exchange(
            scope_id,
            {"role": u.role, "content": u.content},
            {"role": a.role, "content": a.content},
        )

    final = turns[-1]
    return coordinator.build_prompt(scope_id, [{"role": final.role, "content": final.content}])


def _format_messages(messages: list[dict[str, str]]) -> list[str]:
    out = []
    for m in messages:
        body = m["content"].replace("\n", " ⏎ ")
        if len(body) > 110:
            body = body[:107] + "..."
        out.append(f"  {m['role']}: {body}")
    return out


def run(mode: MemoryMode, recent_window: int = 4) -> str:
    """Run the full scripted convo through a real MemoryCoordinator and
    return a formatted view of the prompt the next call would receive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coord = _make_coordinator(mode, Path(tmpdir) / "harness.db", recent_window)
        scope = memory_scope_id("alice", "scripted")
        result = _replay_pairs_then_recall(coord, scope, SCRIPTED_TURNS)

    lines = [
        f"=== Memory mode: {mode.value} (recent_window={recent_window}) ===",
        f"prompt char count: {result.total_chars} "
        f"(summary={result.summary_chars}, recent={result.recent_chars}, raw_history={result.raw_history_chars})",
        f"prompt message count: {len(result.messages)}",
        "prompt messages:",
        *_format_messages(result.messages),
    ]
    return "\n".join(lines)


def run_diff(recent_window: int = 4) -> str:
    """Side-by-side: run all 3 modes on the same scripted convo and print
    each mode's resulting prompt + char count, so the diff is obvious."""
    blocks = [run(mode, recent_window=recent_window) for mode in MemoryMode]
    return "\n\n".join(blocks)


def run_growth(recent_window: int = 4, max_turns: int = 30) -> str:
    """Demonstrate prompt-size growth as the conversation lengthens.

    Persists `max_turns` synthetic user/assistant pairs and snapshots the
    prompt char count after each pair for both `raw` and `summary` modes.
    Shows that `raw` grows ~linearly while `summary` stays bounded.
    """
    snapshots: list[tuple[int, int, int]] = []  # (turn, raw_chars, summary_chars)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw = _make_coordinator(MemoryMode.RAW, Path(tmpdir) / "raw.db", recent_window)
        summ = _make_coordinator(MemoryMode.SUMMARY, Path(tmpdir) / "summary.db", recent_window)
        scope = memory_scope_id("alice", "growth")

        for i in range(1, max_turns + 1):
            u = {"role": "user", "content": f"Turn {i}: tell me about topic-{i} (paragraph of detail goes here)."}
            a = {
                "role": "assistant",
                "content": f"Topic-{i} response, with sufficiently long body text to exercise context growth.",
            }
            raw.persist_exchange(scope, u, a)
            summ.persist_exchange(scope, u, a)

            recall = [{"role": "user", "content": "What was turn 1 about?"}]
            raw_prompt = raw.build_prompt(scope, recall)
            summ_prompt = summ.build_prompt(scope, recall)
            snapshots.append((i, raw_prompt.total_chars, summ_prompt.total_chars))

    lines = [
        f"=== Prompt-size growth (recent_window={recent_window}, turns up to {max_turns}) ===",
        f"{'turn':>4}  {'raw_chars':>10}  {'summary_chars':>14}  {'ratio (raw/summary)':>20}",
    ]
    for t, r, s in snapshots:
        ratio = f"{r / s:.2f}x" if s > 0 else "n/a"
        lines.append(f"{t:>4}  {r:>10}  {s:>14}  {ratio:>20}")
    return "\n".join(lines)


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
            summary_model_str="harness:deterministic",
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
        choices=["context", "diff", "growth", "user-scope", "all"],
        default="diff",
        help=(
            "diff = side-by-side per-mode prompts on the scripted convo (default); "
            "context = single-mode view (use --mode); "
            "growth = prompt-size growth over many turns (raw vs summary); "
            "user-scope = verify per-user sharing/isolation; "
            "all = run every demo back-to-back."
        ),
    )
    parser.add_argument("--mode", default="all", help="none, raw, summary, or all (used with --demo context)")
    parser.add_argument("--recent-window", type=int, default=4, help="Recent-window size for summary mode")
    parser.add_argument("--max-turns", type=int, default=30, help="Number of turns for --demo growth")
    args = parser.parse_args()

    if args.demo == "user-scope":
        print(run_user_scope_demo())
        return

    if args.demo == "diff":
        print(run_diff(recent_window=args.recent_window))
        return

    if args.demo == "growth":
        print(run_growth(recent_window=args.recent_window, max_turns=args.max_turns))
        return

    if args.demo == "all":
        outputs = [
            run_diff(recent_window=args.recent_window),
            run_growth(recent_window=args.recent_window, max_turns=args.max_turns),
            run_user_scope_demo(),
        ]
        print("\n\n".join(outputs))
        return

    # demo == context
    if args.mode == "all":
        outputs = [run(mode, recent_window=args.recent_window) for mode in MemoryMode]
        print("\n\n".join(outputs))
        return

    mode = parse_memory_mode(args.mode)
    print(run(mode, recent_window=args.recent_window))


if __name__ == "__main__":
    main()
