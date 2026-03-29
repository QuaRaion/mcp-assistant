"""
Streamlit UI — главный файл приложения.

Страницы:
  Чат           — основной чат с ассистентом
  MCP Сервера   — управление MCP серверами (добавить/удалить/тестировать)
  История       — история диалогов
"""

import json
import logging
import streamlit as st

import database as db
from mcp_client import MCPClient
from agents.supervisor_agent import SupervisorAgent
from agents.builtin_search import BUILTIN_SERVER_NAME

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="MCP Assistant", layout="wide")

db.init_db()

if "supervisor" not in st.session_state:
    st.session_state.supervisor = SupervisorAgent()

if "workers_loaded" not in st.session_state:
    st.session_state.workers_loaded = False

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "page" not in st.session_state:
    st.session_state.page = "chat"


def get_sup() -> SupervisorAgent:
    return st.session_state.supervisor


def reload_workers(force=False):
    get_sup().load_workers(force_refresh=force)
    st.session_state.workers_loaded = True


def set_page(page: str, chat_id: int = None):
    st.session_state.page = page
    if chat_id is not None:
        st.session_state.active_chat_id = chat_id


# Sidebar

with st.sidebar:
    st.title("Интеллектуальный ассистент")
    st.caption("ИИ ассистент с интеграцией MCP")
    st.divider()

    # Навигация
    if st.button("💬 Chats", use_container_width=True,
                 type="primary" if st.session_state.page == "chat" else "secondary"):
        set_page("chat")
        st.rerun()

    if st.button("🔌 Servers", use_container_width=True,
                 type="primary" if st.session_state.page == "servers" else "secondary"):
        set_page("servers")
        st.rerun()

    st.divider()

    # Список чатов
    if st.session_state.page == "chat":
        if st.button("➕ New Chat", use_container_width=True):
            chat_id = db.create_chat("New Chat", [])
            set_page("chat", chat_id)
            st.rerun()

        st.markdown("**Chats:**")
        chats = db.get_chats()
        for chat in chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                is_active = st.session_state.active_chat_id == chat["id"]
                if st.button(
                    f"{'▶ ' if is_active else ''}{chat['title']}",
                    key=f"chat_btn_{chat['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    set_page("chat", chat["id"])
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_chat_{chat['id']}"):
                    db.delete_chat(chat["id"])
                    if st.session_state.active_chat_id == chat["id"]:
                        st.session_state.active_chat_id = None
                    st.rerun()

    st.divider()
    st.caption("LangGraph · LangChain · MCP")


# PAGE: CHAT

if st.session_state.page == "chat":

    if not st.session_state.workers_loaded:
        with st.spinner("Loading MCP servers..."):
            reload_workers()

    # Нет активного чата
    if st.session_state.active_chat_id is None:
        st.markdown("## 💬 MCP Assistant")
        st.markdown("Select a chat from the sidebar or create a new one.")

        if st.button("➕ Create first chat", type="primary"):
            chat_id = db.create_chat("New Chat", [])
            set_page("chat", chat_id)
            st.rerun()
        st.stop()

    chat_id = st.session_state.active_chat_id
    chat = db.get_chat(chat_id)

    if not chat:
        st.session_state.active_chat_id = None
        st.rerun()

    # Header + Settings

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"## {chat['title']}")
    with col2:
        show_settings = st.toggle("⚙️ Settings", value=False)

    if show_settings:
        with st.container(border=True):
            st.markdown("### Chat Settings")
            col1, col2 = st.columns(2)

            with col1:
                new_title = st.text_input("Chat name", value=chat["title"])

            with col2:
                # Список всех доступных серверов
                all_servers = get_sup().get_available_server_names()
                current_servers = chat.get("server_names") or []

                # Если пусто — используем все
                if not current_servers:
                    current_servers = all_servers

                selected_servers = st.multiselect(
                    "Active sources",
                    options=all_servers,
                    default=[s for s in current_servers if s in all_servers],
                    help="Which MCP servers this chat can use. Empty = all.",
                )

            col3, col4 = st.columns([1, 1])
            with col3:
                if st.button("💾 Save settings", type="primary"):
                    db.update_chat(
                        chat_id,
                        title=new_title,
                        server_names=selected_servers,
                    )
                    st.success("Saved!")
                    st.rerun()
            with col4:
                if st.button("🗑️ Clear messages"):
                    db.clear_messages(chat_id)
                    st.rerun()

    st.divider()

    # Активные серверы для этого чата
    chat_servers = chat.get("server_names") or []
    available = get_sup().get_available_server_names()
    active_servers = [s for s in chat_servers if s in available] if chat_servers else available

    if active_servers:
        st.caption(f"📡 Sources: {', '.join(active_servers)}")
    else:
        st.caption("⚠️ No sources active for this chat.")

    # Messages

    messages = db.get_all_messages(chat_id)

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                try:
                    meta = json.loads(msg["meta"])
                    used = meta.get("used_servers", [])
                    if used:
                        st.caption(f"📡 {', '.join(used)}")
                except Exception:
                    pass

    # Input

    if prompt := st.chat_input("Ask me anything..."):
        # Сохраняем вопрос
        db.save_message(chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Берём последние 20 сообщений для истории (экономия токенов)
                history = db.get_messages(chat_id, limit=20)
                # Убираем последнее (текущий вопрос который только что добавили)
                history = [m for m in history if m["content"] != prompt or m["role"] != "user"][-20:]

                response = get_sup().chat(
                    user_query=prompt,
                    history=history,
                    allowed_servers=active_servers if chat_servers else None,
                )

            answer = response["answer"]
            used = response.get("used_servers", [])

            st.markdown(answer)
            if used:
                st.caption(f"📡 Sources used: {', '.join(used)}")

        db.save_message(chat_id, "assistant", answer, meta={"used_servers": used})

        # Автоназвание чата по первому сообщению
        if chat["title"] == "New Chat" and len(messages) == 0:
            auto_title = prompt[:40] + ("..." if len(prompt) > 40 else "")
            db.update_chat(chat_id, title=auto_title)

        st.rerun()


# PAGE: SERVERS

elif st.session_state.page == "servers":
    st.header("🔌 MCP Servers")

    # Встроенный WebSearch
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 🌐 {BUILTIN_SERVER_NAME} *(built-in)*")
            st.markdown("DuckDuckGo — always available, no setup required.")
        with col2:
            st.success("✅ Active")

    st.divider()
    st.subheader("External MCP Servers")

    with st.expander("➕ Add new server", expanded=False):
        with st.form("add_server_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Name *", placeholder="GitHub, Notion...")
            with col2:
                new_url = st.text_input("MCP Server URL *", placeholder="https://mcp.example.com/mcp")

            col3, col4 = st.columns(2)
            with col3:
                new_api_key = st.text_input("API Key", type="password")
            with col4:
                new_desc = st.text_input("Description")
            submitted = st.form_submit_button("Connect", type="primary")

        if submitted:
            if not new_name or not new_url:
                st.error("Name and URL required.")
            else:
                with st.spinner(f"Testing connection to {new_name}..."):
                    client = MCPClient(new_url, new_api_key)
                    ok, message = client.probe()
                if ok:
                    try:
                        sid = db.add_server(new_name, new_url, new_api_key, new_desc)
                        tools = client.get_tools_with_schema()
                        if tools:
                            db.cache_tools(sid, tools)
                        st.success(f"✅ {new_name} connected! {message}")
                        reload_workers(force=True)
                        st.rerun()
                    except Exception as e:
                        if "UNIQUE constraint" in str(e):
                            st.error(f"Server '{new_name}' already exists.")
                        else:
                            st.error(f"Error: {e}")
                else:
                    st.error(f"❌ Cannot connect: {message}")

    # Server list

    servers = db.get_servers(active_only=False)

    if not servers:
        st.info("No external servers added yet.")
    else:
        for srv in servers:
            is_active = bool(srv["is_active"])
            icon = "🟢" if is_active else "🔴"
            cached_tools = db.get_cached_tools(srv["id"]) or []

            with st.expander(f"{icon} {srv['name']} — `{srv['url']}`"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**URL:** `{srv['url']}`")
                    if srv.get("description"):
                        st.markdown(f"**Description:** {srv['description']}")
                    st.markdown(f"**API Key:** {'`[set]`' if srv.get('api_key') else '`[not set]`'}")
                    if cached_tools:
                        st.markdown(f"**Tools ({len(cached_tools)}):**")
                        for t in cached_tools:
                            st.markdown(f"  • `{t['name']}` — {t.get('description','')[:80]}")
                with col2:
                    if st.button("🔍 Test", key=f"test_{srv['id']}"):
                        with st.spinner("Testing..."):
                            ok, msg = MCPClient(srv["url"], srv.get("api_key","")).probe()
                        st.success(msg) if ok else st.error(msg)

                    toggle = "⏸ Disable" if is_active else "▶ Enable"
                    if st.button(toggle, key=f"tog_{srv['id']}"):
                        db.update_server(srv["id"], is_active=0 if is_active else 1)
                        reload_workers(force=True)
                        st.rerun()

                    if st.button("🗑️ Delete", key=f"del_{srv['id']}", type="secondary"):
                        db.delete_server(srv["id"])
                        get_sup()._workers.pop(srv["name"], None)
                        st.rerun()

                    if st.button("🔄 Tools", key=f"ref_{srv['id']}"):
                        with st.spinner("Fetching..."):
                            try:
                                tools = MCPClient(srv["url"], srv.get("api_key","")).get_tools_with_schema()
                                db.cache_tools(srv["id"], tools)
                                reload_workers(force=True)
                                st.success(f"{len(tools)} tools")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))