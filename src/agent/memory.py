from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from langchain.chat_models import init_chat_model

from agent.memory_mode import MemoryMode

SUMMARY_INSTRUCTION = """You maintain long-term conversation memory.
You are given:
- the prior summary (covering older turns of this user's conversations)
- a batch of older turns that have just fallen out of the live recent window

Update the summary to incorporate these older turns. Keep concise bullets covering only
durable details:
- user identity/profile/preferences
- ongoing goals/projects
- constraints and commitments
- unresolved follow-ups

If new info contradicts old info, keep the newest info.
Do not include transient chit-chat.
Return plain text only.
"""


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str


SummaryUpdater = Callable[[str, list[dict[str, str]], str], str]


def memory_scope_id(user_id: str | None, conversation_id: str | None = None) -> str:
    """Resolve the persistence scope for memory.

    Memory is shared across multiple conversations for the same user.
    If user_id is unavailable, fall back to the conversation-specific scope.
    """
    if user_id:
        return f"user:{user_id}"
    return f"conversation:{conversation_id or 'default'}"


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    scope_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    summarized_through_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def append_message(self, scope_id: str, role: str, content: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (scope_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (scope_id, role, content, _now_iso()),
            )
            return int(cursor.lastrowid)

    def message_count(self, scope_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_messages(self, scope_id: str, limit: int | None = None) -> list[dict[str, str]]:
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE scope_id = ?
                    ORDER BY id ASC
                    """,
                    (scope_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT role, content, id
                        FROM messages
                        WHERE scope_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (scope_id, limit),
                ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_messages_with_ids(self, scope_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE scope_id = ?
                ORDER BY id ASC
                """,
                (scope_id,),
            ).fetchall()
        return [{"id": int(r["id"]), "role": r["role"], "content": r["content"]} for r in rows]

    def get_summary(self, scope_id: str) -> str:
        return self.get_summary_state(scope_id)[0]

    def get_summary_state(self, scope_id: str) -> tuple[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT summary, summarized_through_id
                FROM summaries
                WHERE scope_id = ?
                """,
                (scope_id,),
            ).fetchone()
        if not row:
            return ("", 0)
        return (row["summary"], int(row["summarized_through_id"]))

    def upsert_summary(
        self,
        scope_id: str,
        summary: str,
        summarized_through_id: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (scope_id, summary, summarized_through_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scope_id)
                DO UPDATE SET
                    summary = excluded.summary,
                    summarized_through_id = excluded.summarized_through_id,
                    updated_at = excluded.updated_at
                """,
                (scope_id, summary, summarized_through_id, _now_iso()),
            )


def default_summary_updater(
    existing_summary: str,
    new_messages: list[dict[str, str]],
    model_str: str,
) -> str:
    model = init_chat_model(model_str)

    older_turns = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in new_messages
    )

    response = model.invoke(
        [
            {"role": "system", "content": SUMMARY_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"Current summary:\n{existing_summary or '(empty)'}\n\n"
                    f"Older turns falling out of the recent window:\n{older_turns}\n\n"
                    "Return updated summary."
                ),
            },
        ]
    )
    return response.content.strip()


@dataclass(frozen=True)
class PromptBuildResult:
    """Result of build_prompt_messages with diagnostics for harness/eval use."""
    messages: list[dict[str, str]]
    summary_chars: int
    recent_chars: int
    raw_history_chars: int

    @property
    def total_chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)


class MemoryCoordinator:
    def __init__(
        self,
        mode: MemoryMode,
        store: MemoryStore,
        summary_model_str: str,
        summary_updater: SummaryUpdater = default_summary_updater,
        summary_recent_window: int = 6,
    ):
        self.mode = mode
        self.store = store
        self.summary_model_str = summary_model_str
        self.summary_updater = summary_updater
        self.summary_recent_window = summary_recent_window

    def build_prompt_messages(
        self,
        scope_id: str,
        incoming_messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        return self.build_prompt(scope_id, incoming_messages).messages

    def build_prompt(
        self,
        scope_id: str,
        incoming_messages: list[dict[str, str]],
    ) -> PromptBuildResult:
        if self.mode == MemoryMode.NONE:
            return PromptBuildResult(
                messages=incoming_messages,
                summary_chars=0,
                recent_chars=0,
                raw_history_chars=0,
            )

        latest_user = _latest_user_message(incoming_messages)
        if latest_user is None:
            return PromptBuildResult(
                messages=incoming_messages,
                summary_chars=0,
                recent_chars=0,
                raw_history_chars=0,
            )

        if self.mode == MemoryMode.RAW:
            history = self.store.get_messages(scope_id)
            raw_chars = sum(len(m["content"]) for m in history)
            return PromptBuildResult(
                messages=history + [latest_user],
                summary_chars=0,
                recent_chars=0,
                raw_history_chars=raw_chars,
            )

        # SUMMARY mode: non-overlapping summary + recent window + latest user.
        summary, _ = self.store.get_summary_state(scope_id)
        recent = self.store.get_messages(scope_id, limit=self.summary_recent_window)

        prompt: list[dict[str, str]] = []
        summary_chars = 0
        if summary:
            summary_block = "Conversation memory summary:\n" + summary
            prompt.append({"role": "system", "content": summary_block})
            summary_chars = len(summary_block)
        prompt.extend(recent)
        prompt.append(latest_user)
        recent_chars = sum(len(m["content"]) for m in recent)
        return PromptBuildResult(
            messages=prompt,
            summary_chars=summary_chars,
            recent_chars=recent_chars,
            raw_history_chars=0,
        )

    def persist_exchange(
        self,
        scope_id: str,
        user_message: dict[str, str],
        assistant_message: dict[str, str],
    ) -> None:
        if self.mode == MemoryMode.NONE:
            return

        self.store.append_message(scope_id, user_message["role"], user_message["content"])
        self.store.append_message(
            scope_id,
            assistant_message["role"],
            assistant_message["content"],
        )

        if self.mode == MemoryMode.SUMMARY:
            self._maybe_fold_into_summary(scope_id)

    def _maybe_fold_into_summary(self, scope_id: str) -> None:
        all_messages = self.store.get_messages_with_ids(scope_id)
        if len(all_messages) <= self.summary_recent_window:
            # Nothing has yet fallen out of the recent window.
            return

        existing_summary, summarized_through_id = self.store.get_summary_state(scope_id)
        out_of_window = all_messages[: -self.summary_recent_window]
        new_to_fold = [m for m in out_of_window if m["id"] > summarized_through_id]
        if not new_to_fold:
            return

        payload = [{"role": m["role"], "content": m["content"]} for m in new_to_fold]
        try:
            updated_summary = self.summary_updater(
                existing_summary,
                payload,
                self.summary_model_str,
            )
        except Exception:
            # Persisted messages stay; skip this summary update so the exchange isn't lost.
            return

        self.store.upsert_summary(
            scope_id,
            updated_summary,
            summarized_through_id=new_to_fold[-1]["id"],
        )


def _latest_user_message(messages: list[dict[str, str]]) -> dict[str, str] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return {"role": "user", "content": message.get("content", "")}
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
