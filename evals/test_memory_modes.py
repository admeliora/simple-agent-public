from agent.memory_harness import (
    SCRIPTED_TURNS,
    USER_A_FACT,
    USER_A_ID,
    USER_B_FACT,
    USER_B_ID,
    compute_user_scope_demo,
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


def test_user_scope_demo_each_user_recalls_their_own_fact_in_a_new_conversation():
    """user-a shared 'Japan' in conversation-1; user-a in conversation-2 must
    receive a prompt that contains 'Japan' (so the agent can answer the recall
    question). Symmetric for user-b / 'France'."""
    result = compute_user_scope_demo()
    assert result.user_a_recalls_own_fact, (
        f"{USER_A_ID}'s conversation-2 prompt should contain {USER_A_FACT!r}"
    )
    assert result.user_b_recalls_own_fact, (
        f"{USER_B_ID}'s conversation-2 prompt should contain {USER_B_FACT!r}"
    )


def test_user_scope_demo_neither_user_sees_the_other_users_fact():
    """Cross-user isolation: user-a's prompt must not contain 'France', and
    user-b's must not contain 'Japan'."""
    result = compute_user_scope_demo()
    assert result.user_a_does_not_see_user_b_fact, (
        f"{USER_A_ID}'s prompt leaked {USER_B_FACT!r} from {USER_B_ID}"
    )
    assert result.user_b_does_not_see_user_a_fact, (
        f"{USER_B_ID}'s prompt leaked {USER_A_FACT!r} from {USER_A_ID}"
    )


def test_user_scope_demo_prompt_shape_matches_seed_plus_recall_question():
    """The reconstructed prompt must be exactly the prior exchange followed by
    the recall question — proves we're not just smuggling content in via a
    fuzzy substring match."""
    result = compute_user_scope_demo()
    assert result.user_a_conv2_prompt == [
        {"role": "user", "content": f"My favorite country is {USER_A_FACT}."},
        {"role": "assistant", "content": f"Got it — {USER_A_FACT}."},
        {"role": "user", "content": "What is my favorite country?"},
    ]
    assert result.user_b_conv2_prompt == [
        {"role": "user", "content": f"My favorite country is {USER_B_FACT}."},
        {"role": "assistant", "content": f"Got it — {USER_B_FACT}."},
        {"role": "user", "content": "What is my favorite country?"},
    ]


def test_user_scope_demo_formatted_output_renders_recall_narrative():
    """Smoke-check the CLI-facing output: it should include the headers a
    reader needs to follow the demo (so /demo user-scope stays readable)."""
    output = run_user_scope_demo()
    assert "User-scoped recall demo" in output
    assert "Conversation-1:" in output
    assert "Conversation-2" in output
    assert "Properties verified" in output
    assert "[PASS]" in output and "[FAIL]" not in output
