from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from langchain.chat_models import init_chat_model

from agent.memory_mode import MemoryMode

SUMMARY_INSTRUCTION = """You maintain long-term conversation memory.
Update the memory summary using:
- prior summary
- most recent user+assistant exchange

Keep concise bullets covering only durable details:
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
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, _now_iso()),
            )

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[dict[str, str]]:
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id ASC
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT role, content, id
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (conversation_id, limit),
                ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_summary(self, conversation_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT summary
                FROM summaries
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def upsert_summary(self, conversation_id: str, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (conversation_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id)
                DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at
                """,
                (conversation_id, summary, _now_iso()),
            )


def default_summary_updater(
    existing_summary: str,
    new_messages: list[dict[str, str]],
    model_str: str,
) -> str:
    model = init_chat_model(model_str)

    recent_exchange = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in new_messages
    )

    response = model.invoke(
        [
            {"role": "system", "content": SUMMARY_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"Current summary:\n{existing_summary or '(empty)'}\n\n"
                    f"Recent exchange:\n{recent_exchange}\n\n"
                    "Return updated summary."
                ),
            },
        ]
    )
    return response.content.strip()


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
        conversation_id: str,
        incoming_messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if self.mode == MemoryMode.NONE:
            return incoming_messages

        latest_user = _latest_user_message(incoming_messages)
        if latest_user is None:
            return incoming_messages

        if self.mode == MemoryMode.RAW:
            history = self.store.get_messages(conversation_id)
            return history + [latest_user]

        summary = self.store.get_summary(conversation_id)
        recent = self.store.get_messages(conversation_id, limit=self.summary_recent_window)

        prompt_messages: list[dict[str, str]] = []
        if summary:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": "Conversation memory summary:\n" + summary,
                }
            )
        prompt_messages.extend(recent)
        prompt_messages.append(latest_user)
        return prompt_messages

    def persist_exchange(
        self,
        conversation_id: str,
        user_message: dict[str, str],
        assistant_message: dict[str, str],
    ) -> None:
        if self.mode == MemoryMode.NONE:
            return

        self.store.append_message(conversation_id, user_message["role"], user_message["content"])
        self.store.append_message(
            conversation_id,
            assistant_message["role"],
            assistant_message["content"],
        )

        if self.mode == MemoryMode.SUMMARY:
            existing_summary = self.store.get_summary(conversation_id)
            updated = self.summary_updater(
                existing_summary,
                [user_message, assistant_message],
                self.summary_model_str,
            )
            self.store.upsert_summary(conversation_id, updated)


def _latest_user_message(messages: list[dict[str, str]]) -> dict[str, str] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return {"role": "user", "content": message.get("content", "")}
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
