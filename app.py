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

if "chat_messages" not in st.session_state:
    history = db.get_history(limit=100)
    st.session_state.chat_messages = [
        {"role": m["role"], "content": m["content"]} for m in history
    ]


def get_supervisor() -> SupervisorAgent:
    return st.session_state.supervisor


def reload_workers(force: bool = False):
    sup = get_supervisor()
    summary = sup.load_workers(force_refresh=force)
    st.session_state.workers_loaded = True
    return summary


# Боковая панель навигации и информации о подключённых серверах:

with st.sidebar:
    st.title("Интеллектуальный ассистент")
    st.caption("ИИ ассистент с интеграцией MCP")
    st.divider()

    page = st.radio(
        "Navigation",
        ["💬 Chat", "🔌 Servers", "📜 History"],
        index=0,
    )

    st.divider()

    # Показываем подключённые серверы
    servers = db.get_servers()
    if servers:
        st.markdown(f"**🔌 Подключено: ({len(servers)}):**")
        for s in servers:
            st.markdown(f"  • `{s['name']}`")
    else:
        st.caption("Пока что нет подключенных внешних серверов.\nПерейдите **🔌 MCP Сервера** чтобы добавить")

    st.divider()
    st.caption("ГУАП 2026, Махкамов Шерзод")


# PAGE: CHAT

if page == "💬 Chat":
    st.header("💬 Chat")

    if not st.session_state.workers_loaded:
        with st.spinner("Loading sources..."):
            reload_workers()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            with st.spinner("Refreshing..."):
                reload_workers(force=True)
            st.success("Reloaded!")
    with col3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_messages = []
            db.clear_history()
            st.rerun()

    st.divider()

    # Статус источников
    sup = get_supervisor()
    active_workers = list(sup._workers.keys())
    external = [w for w in active_workers if w != BUILTIN_SERVER_NAME]

    if external:
        st.info(f"🔌 Sources: **{BUILTIN_SERVER_NAME}** + {', '.join(f'**{e}**' for e in external)}")
    else:
        st.info(f"🌐 Using built-in **WebSearch**. Add external servers in **🔌 Servers**.")

    # Чат
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("used_servers"):
                st.caption(f"📡 Sources used: {', '.join(msg['used_servers'])}")

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        db.save_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = sup.chat(prompt)

            answer = response["answer"]
            used = response.get("used_servers", [])

            st.markdown(answer)
            if used:
                st.caption(f"📡 Sources used: {', '.join(used)}")

        msg_data = {"role": "assistant", "content": answer, "used_servers": used}
        st.session_state.chat_messages.append(msg_data)
        db.save_message("assistant", answer, meta={"used_servers": used})


# PAGE: SERVERS

elif page == "🔌 Servers":
    st.header("🔌 MCP Servers")

    # Инфо о встроенном WebSearch
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 🌐 {BUILTIN_SERVER_NAME} *(built-in)*")
            st.markdown("DuckDuckGo web search — **always available**, no setup required.")
            st.markdown("**Tools:** `web_search` — search the internet for any topic")
        with col2:
            st.markdown("")
            st.success("✅ Active")

    st.divider()
    st.subheader("External MCP Servers")
    st.caption("Connect to any MCP server: enter its URL and optional API key.")

    # Add server

    with st.expander("➕ Add new MCP server", expanded=False):
        with st.form("add_server_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Server name *", placeholder="e.g. Notion, GitHub")
            with col2:
                new_url = st.text_input("MCP Server URL *", placeholder="https://your-mcp-server.com/mcp")

            col3, col4 = st.columns(2)
            with col3:
                new_api_key = st.text_input("API Key (optional)", type="password")
            with col4:
                new_desc = st.text_input("Description (optional)", placeholder="What data is here?")

            submitted = st.form_submit_button("Connect", type="primary")

        if submitted:
            if not new_name or not new_url:
                st.error("Name and URL are required.")
            else:
                with st.spinner(f"Testing connection to {new_name}..."):
                    client = MCPClient(new_url, new_api_key)
                    ok, message = client.probe()

                if ok:
                    try:
                        server_id = db.add_server(new_name, new_url, new_api_key, new_desc)
                        tools = client.get_tools_with_schema()
                        if tools:
                            db.cache_tools(server_id, tools)
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
            status_icon = "🟢" if is_active else "🔴"
            cached_tools = db.get_cached_tools(srv["id"]) or []

            with st.expander(f"{status_icon} {srv['name']} — `{srv['url']}`", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**URL:** `{srv['url']}`")
                    if srv.get("description"):
                        st.markdown(f"**Description:** {srv['description']}")
                    st.markdown(f"**API Key:** {'`[set]`' if srv.get('api_key') else '`[not set]`'}")
                    st.markdown(f"**Added:** {srv['created_at']}")
                    if cached_tools:
                        st.markdown(f"**Tools ({len(cached_tools)}):**")
                        for t in cached_tools:
                            desc = t.get("description", "")[:80]
                            st.markdown(f"  • `{t['name']}` — {desc}")
                    else:
                        st.markdown("**Tools:** not loaded")

                with col2:
                    if st.button("🔍 Test", key=f"test_{srv['id']}"):
                        with st.spinner("Testing..."):
                            client = MCPClient(srv["url"], srv.get("api_key", ""))
                            ok, msg = client.probe()
                        if ok:
                            st.success(msg)
                            tools = client.get_tools_with_schema()
                            if tools:
                                db.cache_tools(srv["id"], tools)
                        else:
                            st.error(msg)

                    toggle_label = "⏸ Disable" if is_active else "▶ Enable"
                    if st.button(toggle_label, key=f"toggle_{srv['id']}"):
                        db.update_server(srv["id"], is_active=0 if is_active else 1)
                        reload_workers(force=True)
                        st.rerun()

                    if st.button("🗑️ Delete", key=f"del_{srv['id']}", type="secondary"):
                        db.delete_server(srv["id"])
                        sup = get_supervisor()
                        sup._workers.pop(srv["name"], None)
                        st.rerun()

                    if st.button("🔄 Tools", key=f"refresh_{srv['id']}"):
                        with st.spinner("Fetching..."):
                            try:
                                client = MCPClient(srv["url"], srv.get("api_key", ""))
                                tools = client.get_tools_with_schema()
                                db.cache_tools(srv["id"], tools)
                                reload_workers(force=True)
                                st.success(f"{len(tools)} tools")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))


# Страница "История"
elif page == "📜 History":
    st.header("История чата")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear all", type="secondary"):
            db.clear_history()
            st.session_state.chat_messages = []
            st.success("Cleared.")
            st.rerun()

    history = db.get_history(limit=200)

    if not history:
        st.info("No history yet. Start chatting!")
    else:
        st.caption(f"{len(history)} messages")
        st.divider()
        for msg in history:
            role = msg["role"]
            icon = "🧑" if role == "user" else "🤖"
            time = msg.get("created_at", "")[:16]
            st.markdown(f"**{icon} {role.capitalize()}** `{time}`")
            st.markdown(msg["content"])
            if msg.get("meta"):
                try:
                    meta = json.loads(msg["meta"])
                    used = meta.get("used_servers", [])
                    if used:
                        st.caption(f"📡 {', '.join(used)}")
                except Exception:
                    pass
            st.divider()
