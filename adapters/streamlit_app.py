"""Streamlit адаптер с авторизацией и приватными MCP серверами."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, logging
import streamlit as st
import database as db
from mcp_client import MCPClient
from core.assistant import (
    get_supervisor, reload_workers, invalidate_supervisor,
    get_or_create_active_chat, create_new_chat, list_chats,
    get_chat, delete_chat, clear_chat, update_chat_settings,
    process_message, get_chat_history, get_available_servers,
)
from agents.builtin_search import BUILTIN_SERVER_NAME
from agents.supervisor_agent import RECENT_MESSAGES_WINDOW

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="MCP Assistant", layout="wide")
db.init_db()

def is_logged_in():
    return st.session_state.get("user_id") is not None

def uid():
    return st.session_state["user_id"]

# ── AUTH ──────────────────────────────────────────────────────────────────────
def page_login():
    st.markdown("## Вход в MCP Assistant")
    tab1, tab2 = st.tabs(["Войти", "Регистрация"])
    with tab1:
        with st.form("lf"):
            u = st.text_input("Логин")
            p = st.text_input("Пароль", type="password")
            if st.form_submit_button("Войти", type="primary"):
                user = db.authenticate_user(u, p)
                if user:
                    st.session_state.update({"user_id": user["id"], "username": user["username"],
                                             "active_chat_id": None, "workers_loaded": False})
                    st.rerun()
                else:
                    st.error("Неверный логин или пароль")
    with tab2:
        with st.form("rf"):
            u2 = st.text_input("Логин")
            p2 = st.text_input("Пароль", type="password")
            p3 = st.text_input("Повторите пароль", type="password")
            if st.form_submit_button("Зарегистрироваться", type="primary"):
                if not u2 or not p2:
                    st.error("Заполните все поля")
                elif p2 != p3:
                    st.error("Пароли не совпадают")
                elif len(p2) < 4:
                    st.error("Минимум 4 символа")
                else:
                    if db.create_user(u2, p2):
                        st.success("Аккаунт создан! Войдите.")
                    else:
                        st.error("Логин уже занят")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    user_id = uid()

    if not st.session_state.get("workers_loaded"):
        with st.spinner("Загрузка MCP серверов..."):
            reload_workers(user_id)
        st.session_state["workers_loaded"] = True

    with st.sidebar:
        st.title("MCP Assistant")
        st.caption(f"👤 {st.session_state.get('username','')}")
        if st.button("Выйти", use_container_width=True):
            for k in ["user_id","username","active_chat_id","workers_loaded","page","show_settings"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()
        page = st.session_state.get("page","chat")
        if st.button("💬 Чаты", use_container_width=True, type="primary" if page=="chat" else "secondary"):
            st.session_state["page"] = "chat"; st.rerun()
        if st.button("🔌 MCP сервера", use_container_width=True, type="primary" if page=="servers" else "secondary"):
            st.session_state["page"] = "servers"; st.rerun()
        st.divider()
        if page == "chat":
            if st.button("➕ Новый чат", use_container_width=True, type="primary"):
                chat = create_new_chat(user_id)
                st.session_state["active_chat_id"] = chat["id"]; st.rerun()
            st.markdown("### Чаты")
            for chat in list_chats(user_id):
                c1, c2 = st.columns([4,1])
                is_active = st.session_state.get("active_chat_id") == chat["id"]
                with c1:
                    if st.button(f"{'▶ ' if is_active else ''}{chat['title']}", key=f"cb_{chat['id']}",
                                 use_container_width=True, type="primary" if is_active else "secondary"):
                        st.session_state["active_chat_id"] = chat["id"]
                        st.session_state.pop("show_settings",None); st.rerun()
                with c2:
                    if st.button("🗑", key=f"cd_{chat['id']}"):
                        delete_chat(chat["id"], user_id)
                        if st.session_state.get("active_chat_id") == chat["id"]:
                            st.session_state["active_chat_id"] = None
                        st.rerun()

    if st.session_state.get("page","chat") == "chat":
        _page_chat(user_id)
    else:
        _page_servers(user_id)


def _page_chat(user_id):
    if not st.session_state.get("active_chat_id"):
        chat = get_or_create_active_chat(user_id)
        st.session_state["active_chat_id"] = chat["id"]

    chat_id = st.session_state["active_chat_id"]
    chat = get_chat(chat_id, user_id)
    if not chat:
        st.session_state["active_chat_id"] = None; st.rerun()

    c1, c2 = st.columns([5,1])
    with c1:
        st.markdown(f"## {chat['title']}")
    with c2:
        if st.button("⚙️ Настройки"):
            st.session_state["show_settings"] = not st.session_state.get("show_settings", False)

    if st.session_state.get("show_settings"):
        with st.container(border=True):
            st.markdown("### Настройки чата")
            c1, c2 = st.columns(2)
            with c1:
                new_title = st.text_input("Название", value=chat["title"])
            with c2:
                all_servers = get_available_servers(user_id)
                cur_srv = chat.get("server_names") or all_servers
                selected = st.multiselect("MCP серверы", options=all_servers,
                                          default=[s for s in cur_srv if s in all_servers])
            c3, c4 = st.columns(2)
            with c3:
                if st.button("💾 Сохранить", type="primary"):
                    update_chat_settings(chat_id, user_id, title=new_title, server_names=selected)
                    st.success("Сохранено!"); st.rerun()
            with c4:
                if st.button("🗑️ Очистить"):
                    clear_chat(chat_id, user_id); st.rerun()

    st.divider()
    summary_row = db.get_summary(chat_id)
    if summary_row:
        with st.expander("🧠 Резюме диалога", expanded=False):
            st.caption(summary_row["summary"])

    for msg in get_chat_history(chat_id, user_id):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                try:
                    meta = json.loads(msg["meta"])
                    if meta.get("used_servers"):
                        st.caption(f"📡 {', '.join(meta['used_servers'])}")
                    if meta.get("needs_clarification"):
                        st.caption("❓ Требуется уточнение")
                except Exception:
                    pass

    if prompt := st.chat_input("Задайте вопрос..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Обрабатываю..."):
                result = process_message(user_id, chat_id, prompt)
            st.markdown(result["answer"])
            if result["used_servers"]:
                st.caption(f"📡 {', '.join(result['used_servers'])}")
            if result["needs_clarification"]:
                st.caption("❓ Уточните запрос")
        st.rerun()


def _page_servers(user_id):
    st.header("🔌 MCP сервера")
    with st.container(border=True):
        c1, c2 = st.columns([4,1])
        with c1:
            st.markdown(f"### 🌐 {BUILTIN_SERVER_NAME}")
            st.markdown("DuckDuckGo — встроенный поиск. Всегда активен.")
        with c2:
            st.success("Активен")

    st.divider()
    st.subheader("Мои MCP серверы")

    with st.expander("➕ Добавить", expanded=False):
        with st.form("asf", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                n = st.text_input("Название*")
            with c2:
                u = st.text_input("URL*")
            c3, c4 = st.columns(2)
            with c3:
                k = st.text_input("API Key", type="password")
            with c4:
                d = st.text_input("Описание")
            if st.form_submit_button("Подключить", type="primary"):
                if not n or not u:
                    st.error("Название и URL обязательны")
                else:
                    with st.spinner("Проверка..."):
                        client = MCPClient(u, k)
                        ok, msg = client.probe()
                    if ok:
                        try:
                            sid = db.add_server(user_id, n, u, k, d)
                            tools = client.get_tools_with_schema()
                            if tools:
                                db.cache_tools(sid, user_id, tools)
                            st.success(f"✅ {n} подключён! {msg}")
                            reload_workers(user_id, force=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка: {'Сервер уже существует' if 'UNIQUE' in str(e) else e}")
                    else:
                        st.error(f"❌ {msg}")

    servers = db.get_servers(user_id, active_only=False)
    if not servers:
        st.info("Нет подключённых серверов.")
    else:
        for srv in servers:
            is_active = bool(srv["is_active"])
            cached = db.get_cached_tools(srv["id"]) or []
            with st.expander(f"{'🟢' if is_active else '🔴'} {srv['name']} — `{srv['url']}`"):
                c1, c2 = st.columns([3,1])
                with c1:
                    st.markdown(f"**URL:** `{srv['url']}`")
                    if srv.get("description"):
                        st.markdown(f"**Описание:** {srv['description']}")
                    st.markdown(f"**API Key:** {'`[задан]`' if srv.get('api_key') else '`[не задан]`'}")
                    if cached:
                        st.markdown(f"**Инструменты ({len(cached)}):**")
                        for t in cached:
                            st.markdown(f"  • `{t['name']}` — {t.get('description','')[:80]}")
                with c2:
                    if st.button("🔍 Тест", key=f"t_{srv['id']}"):
                        with st.spinner("..."):
                            ok, msg = MCPClient(srv["url"], srv.get("api_key","")).probe()
                        st.success(msg) if ok else st.error(msg)
                    if st.button("⏸ Стоп" if is_active else "▶ Вкл", key=f"tog_{srv['id']}"):
                        db.update_server(srv["id"], user_id, is_active=0 if is_active else 1)
                        reload_workers(user_id, force=True); st.rerun()
                    if st.button("🗑️ Удалить", key=f"del_{srv['id']}", type="secondary"):
                        db.delete_server(srv["id"], user_id)
                        invalidate_supervisor(user_id)
                        reload_workers(user_id, force=True); st.rerun()
                    if st.button("🔄 Обновить", key=f"ref_{srv['id']}"):
                        with st.spinner("..."):
                            try:
                                tools = MCPClient(srv["url"], srv.get("api_key","")).get_tools_with_schema()
                                db.cache_tools(srv["id"], user_id, tools)
                                reload_workers(user_id, force=True)
                                st.success(f"{len(tools)} tools"); st.rerun()
                            except Exception as e:
                                st.error(str(e))

if not is_logged_in():
    page_login()
else:
    main()
