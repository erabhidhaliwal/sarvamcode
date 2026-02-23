"""
Memory: Persistent conversation storage for Sarvam-OS.
Supports JSON and SQLite backends for cross-session context.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(**data)


class MemoryStore:
    def __init__(
        self,
        project_path: Union[Path, str],
        backend: str = "json",
        max_messages: int = 1000,
    ):
        self.project_path = Path(project_path).resolve()
        self.backend = backend
        self.max_messages = max_messages
        self._messages: list[Message] = []
        self._memory_dir = self.project_path / ".sarvam"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        if backend == "sqlite":
            self._db_path = self._memory_dir / "memory.db"
            self._init_sqlite()
        else:
            self._json_path = self._memory_dir / "memory.json"

        self.load()

    def _init_sqlite(self) -> None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                observation_type TEXT,
                observation_data TEXT,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        """)
        conn.commit()
        conn.close()

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(message)

        if len(self._messages) > self.max_messages:
            removed = self._messages.pop(0)
            if self.backend == "sqlite":
                self._remove_old_sqlite()

        if self.backend == "sqlite":
            self._add_sqlite(message)
        else:
            self._save_json()

        return message

    def add_observation(
        self,
        observation_type: str,
        data: str,
        associated_message_idx: int = -1,
    ) -> None:
        if self._messages:
            msg = self._messages[associated_message_idx]
            msg.metadata["observation"] = {
                "type": observation_type,
                "data": data,
            }
            self.save()

    def get_messages(self, limit: Optional[int] = None) -> list[Message]:
        if limit:
            return self._messages[-limit:]
        return self._messages.copy()

    def get_context_window(self, max_tokens: int = 8000) -> list[dict[str, str]]:
        messages = []
        total_chars = 0
        max_chars = max_tokens * 4

        for msg in reversed(self._messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > max_chars:
                break
            messages.insert(0, {"role": msg.role, "content": msg.content})
            total_chars += msg_chars

        return messages

    def load(self) -> None:
        if self.backend == "sqlite":
            self._load_sqlite()
        else:
            self._load_json()

    def save(self) -> None:
        if self.backend == "sqlite":
            self._save_sqlite()
        else:
            self._save_json()

    def clear(self) -> None:
        self._messages = []
        if self.backend == "sqlite":
            self._clear_sqlite()
        else:
            self._clear_json()

    def _load_json(self) -> None:
        if self._json_path.exists():
            try:
                with open(self._json_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._messages = [Message.from_dict(m) for m in data]
            except (json.JSONDecodeError, KeyError):
                self._messages = []

    def _save_json(self) -> None:
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(
                [m.to_dict() for m in self._messages],
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _clear_json(self) -> None:
        if self._json_path.exists():
            self._json_path.unlink()

    def _add_sqlite(self, message: Message) -> None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (message.role, message.content, message.timestamp, json.dumps(message.metadata)),
        )
        conn.commit()
        conn.close()

    def _load_sqlite(self) -> None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, timestamp, metadata FROM messages ORDER BY id ASC"
        )
        rows = cursor.fetchall()
        conn.close()
        self._messages = [
            Message(
                role=row[0],
                content=row[1],
                timestamp=row[2],
                metadata=json.loads(row[3]) if row[3] else {},
            )
            for row in rows
        ]

    def _save_sqlite(self) -> None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        for msg in self._messages:
            cursor.execute(
                "INSERT INTO messages (role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
                (msg.role, msg.content, msg.timestamp, json.dumps(msg.metadata)),
            )
        conn.commit()
        conn.close()

    def _clear_sqlite(self) -> None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM observations")
        conn.commit()
        conn.close()

    def _remove_old_sqlite(self) -> None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE id IN (SELECT id FROM messages ORDER BY id ASC LIMIT 1)"
        )
        conn.commit()
        conn.close()

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_messages": len(self._messages),
            "backend": self.backend,
            "path": str(self._memory_dir),
            "roles": {
                role: sum(1 for m in self._messages if m.role == role)
                for role in set(m.role for m in self._messages)
            },
        }
