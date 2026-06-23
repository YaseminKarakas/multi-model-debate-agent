from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from .models import DebateResult


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS debate_sessions (
    session_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    model TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    transcript_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class DebateStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(CREATE_TABLE_SQL)
            await db.commit()

    async def save(self, result: DebateResult) -> None:
        payload = result.model_dump(mode="json")
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO debate_sessions (
                    session_id, query, model, final_answer, transcript_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.session_id,
                    result.query,
                    result.model,
                    result.final_answer,
                    json.dumps(payload, indent=2),
                    result.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def recent(self, limit: int = 10) -> list[dict]:
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT session_id, query, model, final_answer, created_at
                FROM debate_sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get(self, session_id: str) -> DebateResult | None:
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT transcript_json
                FROM debate_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return DebateResult.model_validate_json(row["transcript_json"])
