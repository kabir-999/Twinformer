from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel

from app.schemas.workloads import BenchmarkRecord


class BenchmarkRepository:
    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS benchmark_records (id TEXT PRIMARY KEY, source TEXT NOT NULL, payload TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)")

    def add(self, record: BenchmarkRecord) -> None:
        with self._connect() as con:
            con.execute("INSERT OR IGNORE INTO benchmark_records VALUES (?, ?, ?)", (record.id, record.source, record.model_dump_json()))

    def list(self) -> list[BenchmarkRecord]:
        with self._connect() as con:
            rows = con.execute("SELECT payload FROM benchmark_records ORDER BY id").fetchall()
        return [BenchmarkRecord.model_validate(json.loads(row[0])) for row in rows]

    def add_study(self, study: BaseModel) -> None:
        with self._connect() as con:
            con.execute("INSERT OR IGNORE INTO studies VALUES (?, ?, ?, ?)", (study.id, study.kind, study.created_at, study.model_dump_json()))

    def list_studies(self, kind: str | None = None) -> list[dict]:
        query, args = "SELECT payload FROM studies", ()
        if kind:
            query, args = query + " WHERE kind = ?", (kind,)
        with self._connect() as con:
            rows = con.execute(query + " ORDER BY created_at DESC", args).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_chat(self, chat: BaseModel) -> None:
        with self._connect() as con:
            con.execute("INSERT OR REPLACE INTO chats VALUES (?, ?, ?)", (chat.id, chat.updated_at, chat.model_dump_json()))

    def get_chat(self, chat_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT payload FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_chats(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT payload FROM chats ORDER BY updated_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_chat(self, chat_id: str) -> bool:
        with self._connect() as con:
            return con.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount > 0
