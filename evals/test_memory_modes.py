from agent.memory_harness import SCRIPTED_TURNS, build_context, run, run_user_scope_demo
from agent.memory_mode import MemoryMode, parse_memory_mode


def test_parse_memory_mode_defaults_to_none_for_unknown_values():
    assert parse_memory_mode("invalid-mode") == MemoryMode.NONE


def test_parse_memory_mode_parses_valid_values_case_insensitively():
    assert parse_memory_mode("RAW") == MemoryMode.RAW
    assert parse_memory_mode("summary") == MemoryMode.SUMMARY


def test_harness_raw_mode_includes_full_scripted_history():
    context_lines = build_context(MemoryMode.RAW, SCRIPTED_TURNS)
    assert len(context_lines) == len(SCRIPTED_TURNS)
    assert context_lines[0].startswith("user: My name is Alice")


def test_harness_summary_mode_includes_summary_and_latest_turn_only():
    context_lines = build_context(MemoryMode.SUMMARY, SCRIPTED_TURNS)
    assert len(context_lines) == 2
    assert context_lines[0].startswith("memory_summary:")
    assert "User name: Alice" in context_lines[0]
    assert context_lines[1].startswith("recent_user_turn:")


def test_harness_none_mode_uses_only_current_turn():
    output = run(MemoryMode.NONE)
    assert "current_user_turn:" in output
    assert "memory_summary:" not in output


def test_harness_user_scope_demo_reports_sharing_and_isolation():
    output = run_user_scope_demo()
    assert "Seed conversations (persisted):" in output
    assert "conversation-2 / user-a prompt built from persistence:" in output
    assert "conversation-2 / user-b prompt built from persistence:" in output
    assert "shared_across_conversations_for_user_a: True" in output
    assert "isolated_from_user_b_for_user_a: True" in output
    assert "shared_across_conversations_for_user_b: True" in output
    assert "isolated_from_user_a_for_user_b: True" in output
