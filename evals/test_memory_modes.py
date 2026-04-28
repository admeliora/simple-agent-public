from agent.memory_harness import (
    SCRIPTED_TURNS,
    run,
    run_diff,
    run_growth,
    run_user_scope_demo,
)
from agent.memory_mode import MemoryMode, parse_memory_mode


def test_parse_memory_mode_defaults_to_none_for_unknown_values():
    assert parse_memory_mode("invalid-mode") == MemoryMode.NONE


def test_parse_memory_mode_parses_valid_values_case_insensitively():
    assert parse_memory_mode("RAW") == MemoryMode.RAW
    assert parse_memory_mode("summary") == MemoryMode.SUMMARY


def test_harness_none_mode_only_includes_recall_turn():
    output = run(MemoryMode.NONE)
    # NONE just passes through the recall turn (no history, no summary).
    assert "Memory mode: none" in output
    assert "Remind me later" in output
    # No prior scripted turns leak in.
    assert "My name is Alice" not in output


def test_harness_raw_mode_includes_full_scripted_history():
    output = run(MemoryMode.RAW)
    # Every persisted turn should appear in the raw prompt.
    for turn in SCRIPTED_TURNS[:-1]:
        # truncate to 40 chars to avoid the harness display truncation
        assert turn.content[:40] in output, f"missing turn: {turn.content[:40]!r}"


def test_harness_summary_mode_summarizes_older_turns_and_keeps_only_recent_window():
    output = run(MemoryMode.SUMMARY, recent_window=4)
    # The system summary block should be present.
    assert "Conversation memory summary:" in output
    # Earliest turn is older than the window — its raw text must be absent
    # (it should only appear via the summary fact extraction).
    assert "Hi! My name is Alice and I prefer concise answers." not in output
    # But the extracted fact should be in the summary.
    assert "user_name=Alice" in output
    # Most recent paired turn should still appear raw.
    assert "Hakone region near Mt Fuji" in output


def test_harness_diff_runs_all_three_modes_with_char_counts():
    output = run_diff()
    assert "Memory mode: none" in output
    assert "Memory mode: raw" in output
    assert "Memory mode: summary" in output
    assert "prompt char count:" in output


def test_harness_growth_demo_shows_raw_grows_faster_than_summary():
    output = run_growth(recent_window=4, max_turns=20)
    assert "Prompt-size growth" in output
    # The last row's raw should comfortably exceed summary; summary is bounded.
    lines = [line for line in output.splitlines() if line.strip().startswith("20")]
    assert lines, "expected a row for turn 20"
    parts = lines[0].split()
    raw_chars = int(parts[1])
    summary_chars = int(parts[2])
    assert raw_chars > summary_chars * 2


def test_harness_user_scope_demo_reports_sharing_and_isolation():
    output = run_user_scope_demo()
    assert "Seed conversations (persisted):" in output
    assert "conversation-2 / user-a prompt built from persistence:" in output
    assert "conversation-2 / user-b prompt built from persistence:" in output
    assert "shared_across_conversations_for_user_a: True" in output
    assert "isolated_from_user_b_for_user_a: True" in output
    assert "shared_across_conversations_for_user_b: True" in output
    assert "isolated_from_user_a_for_user_b: True" in output
