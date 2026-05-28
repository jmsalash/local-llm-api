"""SQLite persistence for agents, conversation history, and long-term memories.

Kept deliberately simple (stdlib sqlite3). Calls are synchronous; the dataset for a
local dev backend is tiny, and the dominant latency is model generation, not the DB.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            model TEXT NOT NULL,
            directive TEXT NOT NULL DEFAULT '',
            reasoning INTEGER NOT NULL DEFAULT 0,
            temperature REAL NOT NULL DEFAULT 0.7,
            num_ctx INTEGER NOT NULL DEFAULT 8192,
            max_history INTEGER NOT NULL DEFAULT 20,
            top_k INTEGER NOT NULL DEFAULT 5,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            thinking TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_agent_session
            ON messages(agent_id, session_id, created_at);

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
        """
    )
    conn.commit()


# --- agents ---------------------------------------------------------------

def create_agent(
    *,
    name: str,
    model: str,
    directive: str,
    reasoning: bool,
    temperature: float,
    num_ctx: int,
    max_history: int,
    top_k: int,
) -> dict[str, Any]:
    aid = new_id("agent")
    ts = _now()
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO agents
               (id, name, model, directive, reasoning, temperature, num_ctx,
                max_history, top_k, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (aid, name, model, directive, int(reasoning), temperature, num_ctx,
             max_history, top_k, ts, ts),
        )
        conn.commit()
    return get_agent(aid)  # type: ignore[return-value]


def get_agent(agent_id: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM agents WHERE id=?", (agent_id,)).fetchone()
    return _agent_row(row) if row else None


def get_agent_by_name(name: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM agents WHERE name=?", (name,)).fetchone()
    return _agent_row(row) if row else None


def list_agents() -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute("SELECT * FROM agents ORDER BY created_at").fetchall()
    return [_agent_row(r) for r in rows]


def update_agent(agent_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    allowed = {"name", "model", "directive", "reasoning", "temperature",
               "num_ctx", "max_history", "top_k"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return get_agent(agent_id)
    if "reasoning" in sets:
        sets["reasoning"] = int(sets["reasoning"])
    sets["updated_at"] = _now()
    cols = ", ".join(f"{k}=?" for k in sets)
    with _lock:
        conn = get_conn()
        conn.execute(f"UPDATE agents SET {cols} WHERE id=?", (*sets.values(), agent_id))
        conn.commit()
    return get_agent(agent_id)


def delete_agent(agent_id: str) -> bool:
    with _lock:
        conn = get_conn()
        cur = conn.execute("DELETE FROM agents WHERE id=?", (agent_id,))
        conn.execute("DELETE FROM messages WHERE agent_id=?", (agent_id,))
        conn.execute("DELETE FROM memories WHERE agent_id=?", (agent_id,))
        conn.commit()
    return cur.rowcount > 0


def _agent_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["reasoning"] = bool(d["reasoning"])
    return d


# --- messages (conversation history) -------------------------------------

def add_message(agent_id: str, session_id: str, role: str, content: str,
                thinking: str | None = None) -> dict[str, Any]:
    mid = new_id("msg")
    ts = _now()
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO messages (id, agent_id, session_id, role, content, thinking, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (mid, agent_id, session_id, role, content, thinking, ts),
        )
        conn.commit()
    return {"id": mid, "agent_id": agent_id, "session_id": session_id,
            "role": role, "content": content, "thinking": thinking, "created_at": ts}


def get_history(agent_id: str, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    with _lock:
        conn = get_conn()
        if limit:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM messages WHERE agent_id=? AND session_id=?
                       ORDER BY created_at DESC LIMIT ?
                   ) ORDER BY created_at ASC""",
                (agent_id, session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE agent_id=? AND session_id=? ORDER BY created_at ASC",
                (agent_id, session_id),
            ).fetchall()
    return [dict(r) for r in rows]


def list_sessions(agent_id: str) -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute(
            """SELECT session_id, COUNT(*) AS messages, MAX(created_at) AS last_at
               FROM messages WHERE agent_id=? GROUP BY session_id ORDER BY last_at DESC""",
            (agent_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_session(agent_id: str, session_id: str) -> int:
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "DELETE FROM messages WHERE agent_id=? AND session_id=?", (agent_id, session_id)
        )
        conn.commit()
    return cur.rowcount


# --- memories (long-term semantic store) ---------------------------------

def add_memory(agent_id: str, content: str, embedding: list[float]) -> dict[str, Any]:
    mid = new_id("mem")
    ts = _now()
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO memories (id, agent_id, content, embedding, created_at) VALUES (?,?,?,?,?)",
            (mid, agent_id, content, json.dumps(embedding), ts),
        )
        conn.commit()
    return {"id": mid, "agent_id": agent_id, "content": content, "created_at": ts}


def get_memories(agent_id: str) -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute(
            "SELECT id, content, embedding, created_at FROM memories WHERE agent_id=?",
            (agent_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d["embedding"])
        out.append(d)
    return out


def delete_memory(agent_id: str, memory_id: str) -> bool:
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "DELETE FROM memories WHERE id=? AND agent_id=?", (memory_id, agent_id)
        )
        conn.commit()
    return cur.rowcount > 0
