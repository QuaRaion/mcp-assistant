"""
SQLite хранилище.
  - users          — пользователи
  - telegram_users — привязка Telegram → внутренний user_id
  - mcp_servers    — MCP серверы (user_id — у каждого свои)
  - chats          — чаты (user_id)
  - chat_messages  — история
  - chat_summaries — summary memory
"""

import sqlite3
import json
import hashlib
import secrets
from typing import Optional
from config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS telegram_users (
    telegram_id    INTEGER PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    username       TEXT,
    first_name     TEXT,
    active_chat_id INTEGER,
    created_at     TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS mcp_servers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    api_key     TEXT,
    description TEXT,
    tools_cache TEXT,
    is_active   INTEGER DEFAULT 1,
    is_builtin  INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, name)
);
CREATE TABLE IF NOT EXISTS chats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title        TEXT NOT NULL DEFAULT 'Новый чат',
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
CREATE TABLE IF NOT EXISTS chat_summaries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id        INTEGER NOT NULL UNIQUE REFERENCES chats(id) ON DELETE CASCADE,
    summary        TEXT NOT NULL,
    messages_count INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT DEFAULT (datetime('now'))
);
"""

def get_conn():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)

# --- Users ---
def _hash_password(p):
    salt = secrets.token_hex(16)
    return f"{salt}:{hashlib.sha256(f'{salt}{p}'.encode()).hexdigest()}"

def _verify_password(p, stored):
    try:
        salt, h = stored.split(":", 1)
        return hashlib.sha256(f"{salt}{p}".encode()).hexdigest() == h
    except Exception:
        return False

def create_user(username, password):
    try:
        with get_conn() as conn:
            cur = conn.execute("INSERT INTO users (username, password_hash) VALUES (?,?)", (username, _hash_password(password)))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None

def authenticate_user(username, password):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and _verify_password(password, row["password_hash"]):
        return dict(row)
    return None

def get_user(user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None

# --- Telegram ---
def get_or_create_telegram_user(telegram_id, username="", first_name=""):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM telegram_users WHERE telegram_id=?", (telegram_id,)).fetchone()
        if row:
            return dict(row)
        tg_login = username or f"tg_{telegram_id}"
        pw = _hash_password(secrets.token_hex(16))
        try:
            cur = conn.execute("INSERT INTO users (username, password_hash) VALUES (?,?)", (tg_login, pw))
            user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            cur = conn.execute("INSERT INTO users (username, password_hash) VALUES (?,?)", (f"tg_{telegram_id}", pw))
            user_id = cur.lastrowid
        conn.execute("INSERT INTO telegram_users (telegram_id, user_id, username, first_name) VALUES (?,?,?,?)", (telegram_id, user_id, username, first_name))
        return {"telegram_id": telegram_id, "user_id": user_id, "username": username, "first_name": first_name, "active_chat_id": None}

def get_telegram_user(telegram_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM telegram_users WHERE telegram_id=?", (telegram_id,)).fetchone()
    return dict(row) if row else None

def set_telegram_active_chat(telegram_id, chat_id):
    with get_conn() as conn:
        conn.execute("UPDATE telegram_users SET active_chat_id=? WHERE telegram_id=?", (chat_id, telegram_id))

# --- MCP Servers (per user) ---
def add_server(user_id, name, url, api_key="", description="", is_builtin=0):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO mcp_servers (user_id, name, url, api_key, description, is_builtin) VALUES (?,?,?,?,?,?)",
            (user_id, name, url, api_key, description, is_builtin),
        )
        return cur.lastrowid

def update_server(server_id, user_id, **kwargs):
    allowed = {"name", "url", "api_key", "description", "is_active", "tools_cache"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE mcp_servers SET {sets} WHERE id=? AND user_id=?", (*fields.values(), server_id, user_id))

def delete_server(server_id, user_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM mcp_servers WHERE id=? AND user_id=?", (server_id, user_id))

def get_servers(user_id, active_only=True):
    q = "SELECT * FROM mcp_servers WHERE user_id=?"
    params = [user_id]
    if active_only:
        q += " AND is_active=1"
    q += " ORDER BY is_builtin DESC, name"
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]

def get_server(server_id, user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM mcp_servers WHERE id=? AND user_id=?", (server_id, user_id)).fetchone()
    return dict(row) if row else None

def cache_tools(server_id, user_id, tools):
    with get_conn() as conn:
        conn.execute("UPDATE mcp_servers SET tools_cache=? WHERE id=? AND user_id=?",
                     (json.dumps(tools, ensure_ascii=False), server_id, user_id))

def get_cached_tools(server_id):
    with get_conn() as conn:
        row = conn.execute("SELECT tools_cache FROM mcp_servers WHERE id=?", (server_id,)).fetchone()
    if row and row["tools_cache"]:
        return json.loads(row["tools_cache"])
    return None

# --- Chats ---
def create_chat(user_id, title="Новый чат", server_names=None):
    names = json.dumps(server_names or [], ensure_ascii=False)
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO chats (user_id, title, server_names) VALUES (?,?,?)", (user_id, title, names))
        return cur.lastrowid

def get_chats(user_id):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM chats WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["server_names"] = json.loads(d["server_names"])
        result.append(d)
    return result

def get_chat(chat_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["server_names"] = json.loads(d["server_names"])
    return d

def get_chat_safe(chat_id, user_id):
    chat = get_chat(chat_id)
    if chat and chat["user_id"] == user_id:
        return chat
    return None

def update_chat(chat_id, title=None, server_names=None):
    fields = {}
    if title is not None:
        fields["title"] = title
    if server_names is not None:
        fields["server_names"] = json.dumps(server_names, ensure_ascii=False)
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=datetime('now')"
    with get_conn() as conn:
        conn.execute(f"UPDATE chats SET {sets} WHERE id=?", (*fields.values(), chat_id))

def delete_chat(chat_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))

# --- Messages ---
def save_message(chat_id, role, content, meta=None):
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO chat_messages (chat_id, role, content, meta) VALUES (?,?,?,?)",
                           (chat_id, role, content, json.dumps(meta) if meta else None))
        conn.execute("UPDATE chats SET updated_at=datetime('now') WHERE id=?", (chat_id,))
        return cur.lastrowid

def get_messages(chat_id, limit=6):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM chat_messages WHERE chat_id=? ORDER BY id DESC LIMIT ?", (chat_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]

def get_all_messages(chat_id):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM chat_messages WHERE chat_id=? ORDER BY id", (chat_id,)).fetchall()
    return [dict(r) for r in rows]

def count_messages(chat_id):
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM chat_messages WHERE chat_id=?", (chat_id,)).fetchone()
    return row["cnt"] if row else 0

def clear_messages(chat_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_messages WHERE chat_id=?", (chat_id,))
    clear_summary(chat_id)

# --- Summary ---
SUMMARY_EVERY = 10

def get_summary(chat_id):
    with get_conn() as conn:
        row = conn.execute("SELECT summary, messages_count FROM chat_summaries WHERE chat_id=?", (chat_id,)).fetchone()
    return dict(row) if row else None

def save_summary(chat_id, summary, messages_count):
    with get_conn() as conn:
        conn.execute("""INSERT INTO chat_summaries (chat_id, summary, messages_count) VALUES (?,?,?)
               ON CONFLICT(chat_id) DO UPDATE SET summary=excluded.summary,
               messages_count=excluded.messages_count, updated_at=datetime('now')""",
                     (chat_id, summary, messages_count))

def clear_summary(chat_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_summaries WHERE chat_id=?", (chat_id,))

def should_update_summary(chat_id):
    total = count_messages(chat_id)
    row = get_summary(chat_id)
    last = row["messages_count"] if row else 0
    return (total - last) >= SUMMARY_EVERY

def save_tools_hints(server_id, user_id, hints: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE mcp_servers SET tools_hints=? WHERE id=? AND user_id=?",
            (hints, server_id, user_id)
        )

def get_tools_hints(server_id) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT tools_hints FROM mcp_servers WHERE id=?", (server_id,)
        ).fetchone()
    return row["tools_hints"] if row and row["tools_hints"] else ""
