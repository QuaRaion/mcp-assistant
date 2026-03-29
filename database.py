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
    is_builtin  INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL DEFAULT 'New Chat',
    server_names TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    meta       TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# MCP сервера (CRUD + кэш инструментов):

def add_server(name: str, url: str, api_key: str = "", description: str = "", is_builtin: int = 0) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO mcp_servers (name, url, api_key, description, is_builtin) VALUES (?,?,?,?,?)",
            (name, url, api_key, description, is_builtin),
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
    q += " ORDER BY is_builtin DESC, name"
    with get_conn() as conn:
        rows = conn.execute(q).fetchall()
    return [dict(r) for r in rows]

def cache_tools(server_id, tools):
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

# Чаты (CRUD):

def create_chat(title="New Chat", server_names: list[str] = None) -> int:
    names = json.dumps(server_names or [], ensure_ascii=False)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chats (title, server_names) VALUES (?,?)",
            (title, names),
        )
        return cur.lastrowid

def get_chats() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chats ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def get_chat(chat_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["server_names"] = json.loads(d["server_names"])
    return d

def update_chat(chat_id: int, title: str = None, server_names: list[str] = None):
    fields = {}
    if title is not None:
        fields["title"] = title
    if server_names is not None:
        fields["server_names"] = json.dumps(server_names, ensure_ascii=False)
    fields["updated_at"] = "datetime('now')"
    if not fields:
        return
    # updated_at через raw SQL
    sets = ", ".join(
        f"{k}=datetime('now')" if k == "updated_at" else f"{k}=?"
        for k in fields
    )
    values = [v for k, v in fields.items() if k != "updated_at"]
    with get_conn() as conn:
        conn.execute(f"UPDATE chats SET {sets} WHERE id=?", (*values, chat_id))

def delete_chat(chat_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))

# Сообщения в чате:

def save_message(chat_id: int, role: str, content: str, meta: dict = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content, meta) VALUES (?,?,?,?)",
            (chat_id, role, content, json.dumps(meta) if meta else None),
        )
        # Обновляем updated_at чата
        conn.execute(
            "UPDATE chats SET updated_at=datetime('now') WHERE id=?", (chat_id,)
        )
        return cur.lastrowid

def get_messages(chat_id: int, limit: int = 5) -> list[dict]:
    """
    Возвращает последние N сообщений чата.
    limit=5 — оптимально: достаточно контекста, не тратим лишние токены.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM chat_messages
               WHERE chat_id=?
               ORDER BY id DESC LIMIT ?""",
            (chat_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]

def get_all_messages(chat_id: int) -> list[dict]:
    """Все сообщения — для отображения в UI."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE chat_id=? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def clear_messages(chat_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_messages WHERE chat_id=?", (chat_id,))
