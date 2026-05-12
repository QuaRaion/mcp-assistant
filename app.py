"""
Streamlit UI — главный файл приложения.

Страницы:
  Чат           — основной чат с ассистентом
  MCP Сервера   — управление MCP серверами (добавить/удалить/тестировать)
"""

import json
import logging
import streamlit as st

import database as db
from mcp_client import MCPClient
from agents.supervisor_agent import SupervisorAgent, RECENT_MESSAGES_WINDOW
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


# --- Sidebar ---

with st.sidebar:
    st.title("Интеллектуальный ассистент")
    st.caption("ИИ ассистент с интеграцией MCP")
    st.divider()

    if st.button("💬 Чаты", use_container_width=True,
                 type="primary" if st.session_state.page == "chat" else "secondary"):
        set_page("chat")
        st.rerun()

    if st.button("🔌 MCP сервера", use_container_width=True,
                 type="primary" if st.session_state.page == "servers" else "secondary"):
        set_page("servers")
        st.rerun()

    st.divider()

    if st.session_state.page == "chat":
        if st.button("Создать новый чат", use_container_width=True, type="primary"):
            chat_id = db.create_chat("Новый чат", [])
            set_page("chat", chat_id)
            st.rerun()
        st.markdown("## Чаты:")
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


# --- PAGE: CHAT ---

if st.session_state.page == "chat":
    if not st.session_state.workers_loaded:
        with st.spinner("Загрузка MCP серверов..."):
            reload_workers()

    if st.session_state.active_chat_id is None:
        st.markdown("## MCP Assistant")
        st.markdown("Выберите чат в боковой панели или создайте новый")
        if st.button("Создать новый чат", type="primary"):
            chat_id = db.create_chat("Новый чат", [])
            set_page("chat", chat_id)
            st.rerun()
        st.stop()

    chat_id = st.session_state.active_chat_id
    chat = db.get_chat(chat_id)

    if not chat:
        st.session_state.active_chat_id = None
        st.rerun()

    # --- Header + Settings ---

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"## {chat['title']}")
    with col2:
        if "show_settings" not in st.session_state:
            st.session_state.show_settings = False
        if st.button("⚙️ Настройки"):
            st.session_state.show_settings = not st.session_state.show_settings

    if st.session_state.show_settings:
        with st.container(border=True):
            st.markdown("### Настройки чата")
            col1, col2 = st.columns(2)

            with col1:
                new_title = st.text_input("Название чата", value=chat["title"])

            with col2:
                all_servers = get_sup().get_available_server_names()
                current_servers = chat.get("server_names") or []
                if not current_servers:
                    current_servers = all_servers

                selected_servers = st.multiselect(
                    "Активные MCP сервера",
                    options=all_servers,
                    default=[s for s in current_servers if s in all_servers],
                    help="Какие MCP серверы использовать в этом чате. Пустота = все.",
                )

            col3, col4 = st.columns([1, 1])
            with col3:
                if st.button("💾 Сохранить", type="primary"):
                    db.update_chat(chat_id, title=new_title, server_names=selected_servers)
                    st.success("Изменения сохранены!")
                    st.rerun()
            with col4:
                if st.button("🗑️ Очистить чат"):
                    db.clear_messages(chat_id)  # также удаляет резюме
                    st.rerun()

    st.divider()

    # Активные серверы для этого чата
    chat_servers = chat.get("server_names") or []
    available = get_sup().get_available_server_names()
    active_servers = [s for s in chat_servers if s in available] if chat_servers else available

    # --- Отображаем резюме (если есть) ---
    summary_row = db.get_summary(chat_id)
    current_summary = summary_row["summary"] if summary_row else ""
    if current_summary:
        with st.expander("🧠 Резюме диалога", expanded=False):
            st.caption(current_summary)

    # --- Сообщения ---

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

    # --- Ввод ---

    if prompt := st.chat_input("Задайте вопрос..."):
        db.save_message(chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу информацию..."):
                # Загружаем резюме и скользящее окно из БД
                summary_row = db.get_summary(chat_id)
                summary = summary_row["summary"] if summary_row else ""

                # Последние RECENT_MESSAGES_WINDOW сообщений (без только что добавленного вопроса)
                recent = db.get_messages(chat_id, limit=RECENT_MESSAGES_WINDOW + 1)
                recent = [m for m in recent if not (m["role"] == "user" and m["content"] == prompt)]
                recent = recent[-RECENT_MESSAGES_WINDOW:]

                response = get_sup().chat(
                    user_query=prompt,
                    summary=summary,
                    recent_history=recent,
                    allowed_servers=active_servers if chat_servers else None,
                )

            answer = response["answer"]
            used = response.get("used_servers", [])

            st.markdown(answer)
            if used:
                st.caption(f"📡 Использованные сервера: {', '.join(used)}")

        db.save_message(chat_id, "assistant", answer, meta={"used_servers": used})

        # Автоназвание чата по первому сообщению
        if chat["title"] == "Новый чат" and len(messages) == 0:
            auto_title = prompt[:40] + ("..." if len(prompt) > 40 else "")
            db.update_chat(chat_id, title=auto_title)

        # Обновляем резюме в фоне если накопилось достаточно сообщений
        if db.should_update_summary(chat_id):
            with st.spinner("Обновляю память диалога..."):
                get_sup().update_summary(chat_id)

        st.rerun()


# --- PAGE: SERVERS ---

elif st.session_state.page == "servers":
    st.header("🔌 MCP сервера")

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 🌐 {BUILTIN_SERVER_NAME}")
            st.markdown("DuckDuckGo — встроенный инструмент поиска в интернете. Всегда активен.")
        with col2:
            st.success("Активен")

    st.divider()
    st.subheader("Подключенные MCP сервера")

    with st.expander("➕ Добавить новый", expanded=False):
        with st.form("add_server_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Название*", placeholder="GitHub, Notion...")
            with col2:
                new_url = st.text_input("MCP Server URL*", placeholder="https://mcp.example.com/mcp")

            col3, col4 = st.columns(2)
            with col3:
                new_api_key = st.text_input("API Key", type="password")
            with col4:
                new_desc = st.text_input("Описание")
            submitted = st.form_submit_button("Подключиться", type="primary")

        if submitted:
            if not new_name or not new_url:
                st.error("Название и URL обязательны")
            else:
                with st.spinner(f"Тестирование подключения {new_name}..."):
                    client = MCPClient(new_url, new_api_key)
                    ok, message = client.probe()
                if ok:
                    try:
                        sid = db.add_server(new_name, new_url, new_api_key, new_desc)
                        tools = client.get_tools_with_schema()
                        if tools:
                            db.cache_tools(sid, tools)
                        st.success(f"✅ {new_name} подключено! {message}")
                        reload_workers(force=True)
                        st.rerun()
                    except Exception as e:
                        if "UNIQUE constraint" in str(e):
                            st.error(f"Сервер '{new_name}' уже существует.")
                        else:
                            st.error(f"Ошибка: {e}")
                else:
                    st.error(f"❌ Не удалось подключиться: {message}")

    servers = db.get_servers(active_only=False)

    if not servers:
        st.info("Нет подключенных серверов.")
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
                        st.markdown(f"**Описание:** {srv['description']}")
                    st.markdown(f"**API Key:** {'`[задан]`' if srv.get('api_key') else '`[не задан]`'}")
                    if cached_tools:
                        st.markdown(f"**Инструменты ({len(cached_tools)}):**")
                        for t in cached_tools:
                            st.markdown(f"  • `{t['name']}` — {t.get('description', '')[:80]}")
                with col2:
                    if st.button("🔍 Протестировать", key=f"test_{srv['id']}"):
                        with st.spinner("Тестирование..."):
                            ok, msg = MCPClient(srv["url"], srv.get("api_key", "")).probe()
                        st.success(msg) if ok else st.error(msg)

                    toggle = "⏸ Остановить" if is_active else "▶ Включить"
                    if st.button(toggle, key=f"tog_{srv['id']}"):
                        db.update_server(srv["id"], is_active=0 if is_active else 1)
                        reload_workers(force=True)
                        st.rerun()

                    if st.button("🗑️ Удалить", key=f"del_{srv['id']}", type="secondary"):
                        db.delete_server(srv["id"])
                        get_sup()._workers.pop(srv["name"], None)
                        st.rerun()

                    if st.button("🔄 Обновить", key=f"ref_{srv['id']}"):
                        with st.spinner("Получение данных..."):
                            try:
                                tools = MCPClient(srv["url"], srv.get("api_key", "")).get_tools_with_schema()
                                db.cache_tools(srv["id"], tools)
                                reload_workers(force=True)
                                st.success(f"{len(tools)} tools")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
