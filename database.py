"""
SQLite хранилище:
  - mcp_servers   — зарегистрированные MCP серверы пользователя
  - chat_messages — история диалога
"""

import sqlite3
import json
from typing import Optional
from config import DATABASE_PATH


# Схема БД:

SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_servers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    url         TEXT NOT NULL,
    api_key     TEXT,
    description TEXT,
    tools_cache TEXT,          -- JSON список tools (кэш)
    is_active   INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,   -- ['user'|'assistant'|'system']
    content    TEXT NOT NULL,
    meta       TEXT,            -- JSON (какие серверы использовались и т.д.)
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# MCP сервера (CRUD + кэш инструментов):

def add_server(name: str, url: str, api_key: str = "", description: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO mcp_servers (name, url, api_key, description) VALUES (?,?,?,?)",
            (name, url, api_key, description),
        )
        return cur.lastrowid


def update_server(server_id: int, **kwargs) -> None:
    allowed = {"name", "url", "api_key", "description", "is_active", "tools_cache"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE mcp_servers SET {sets} WHERE id=?",
            (*fields.values(), server_id),
        )


def delete_server(server_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM mcp_servers WHERE id=?", (server_id,))


def get_servers(active_only: bool = True) -> list[dict]:
    q = "SELECT * FROM mcp_servers"
    if active_only:
        q += " WHERE is_active=1"
    q += " ORDER BY name"
    with get_conn() as conn:
        rows = conn.execute(q).fetchall()
    return [dict(r) for r in rows]


def get_server_by_id(server_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM mcp_servers WHERE id=?", (server_id,)).fetchone()
    return dict(row) if row else None


def cache_tools(server_id: int, tools: list[dict]) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE mcp_servers SET tools_cache=? WHERE id=?",
            (json.dumps(tools, ensure_ascii=False), server_id),
        )


def get_cached_tools(server_id: int) -> Optional[list[dict]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT tools_cache FROM mcp_servers WHERE id=?", (server_id,)
        ).fetchone()
    if row and row["tools_cache"]:
        return json.loads(row["tools_cache"])
    return None


# История сообщений:

def save_message(role: str, content: str, meta: Optional[dict] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (role, content, meta) VALUES (?,?,?)",
            (role, content, json.dumps(meta) if meta else None),
        )
        return cur.lastrowid


def get_history(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_history() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_messages")
