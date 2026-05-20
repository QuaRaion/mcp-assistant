"""
Ядро системы. Все адаптеры (Streamlit, Telegram) используют только этот модуль.
SupervisorAgent теперь создаётся per-user — у каждого свои воркеры/серверы.
"""

import logging
from typing import Optional
import database as db
from agents.supervisor_agent import SupervisorAgent, RECENT_MESSAGES_WINDOW
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Кэш SupervisorAgent по user_id — один агент на пользователя
_supervisors: dict[int, SupervisorAgent] = {}


def get_supervisor(user_id: int) -> SupervisorAgent:
    """Возвращает SupervisorAgent для конкретного пользователя."""
    if user_id not in _supervisors:
        sup = SupervisorAgent()
        sup.load_workers(user_id=user_id)
        _supervisors[user_id] = sup
    return _supervisors[user_id]


def reload_workers(user_id: int, force: bool = False) -> dict:
    """Перезагружает воркеры пользователя (после добавления/удаления серверов)."""
    sup = get_supervisor(user_id)
    return sup.load_workers(user_id=user_id, force_refresh=force)


def invalidate_supervisor(user_id: int) -> None:
    """Сбрасывает кэш агента пользователя."""
    _supervisors.pop(user_id, None)


# --- Chat management ---

def get_or_create_active_chat(user_id: int) -> dict:
    chats = db.get_chats(user_id)
    if chats:
        return chats[0]
    chat_id = db.create_chat(user_id, "Новый чат")
    return db.get_chat(chat_id)


def create_new_chat(user_id: int, title: str = "Новый чат") -> dict:
    chat_id = db.create_chat(user_id, title)
    return db.get_chat(chat_id)


def list_chats(user_id: int) -> list[dict]:
    return db.get_chats(user_id)


def get_chat(chat_id: int, user_id: int) -> Optional[dict]:
    return db.get_chat_safe(chat_id, user_id)


def delete_chat(chat_id: int, user_id: int) -> bool:
    if not db.get_chat_safe(chat_id, user_id):
        return False
    db.delete_chat(chat_id)
    return True


def clear_chat(chat_id: int, user_id: int) -> bool:
    if not db.get_chat_safe(chat_id, user_id):
        return False
    db.clear_messages(chat_id)
    return True


def update_chat_settings(chat_id: int, user_id: int, title: Optional[str] = None, server_names: Optional[list] = None) -> bool:
    if not db.get_chat_safe(chat_id, user_id):
        return False
    db.update_chat(chat_id, title=title, server_names=server_names)
    return True


# --- Message processing ---

def process_message(user_id: int, chat_id: int, text: str) -> dict:
    """Главная функция — вызывается любым адаптером."""
    chat = db.get_chat_safe(chat_id, user_id)
    if not chat:
        return {"answer": "Чат не найден.", "used_servers": [], "needs_clarification": False, "chat_id": chat_id}

    db.save_message(chat_id, "user", text)
 
    # генерация заголовка чатов через LLM, но может работать медленнее, чем просто обрезка текста:
    if chat["title"] == "Новый чат" and db.count_messages(chat_id) == 1:
        try:
            title_resp = get_supervisor(user_id).llm.invoke([
                SystemMessage(content="Generate a very short chat title (50 symbols max) based on the user's question. No quotes, no punctuation at the end. Reply with title only."),
                HumanMessage(content=text),
            ])
            title = title_resp.content.strip()[:50]
        except Exception:
            title = text[:40] + ("..." if len(text) > 40 else "")
        db.update_chat(chat_id, title=title)
    
    # заголовок - просто обрезка первого запроса пользователя:
    # if chat["title"] == "Новый чат" and db.count_messages(chat_id) == 1:
    #     db.update_chat(chat_id, title=text[:40] + ("..." if len(text) > 40 else ""))
   
    summary_row = db.get_summary(chat_id)
    summary = summary_row["summary"] if summary_row else ""

    recent = db.get_messages(chat_id, limit=RECENT_MESSAGES_WINDOW + 1)
    recent = [m for m in recent if not (m["role"] == "user" and m["content"] == text)]
    recent = recent[-RECENT_MESSAGES_WINDOW:]

    sup = get_supervisor(user_id)
    available = sup.get_available_server_names()
    chat_servers = chat.get("server_names") or []
    allowed = [s for s in chat_servers if s in available] if chat_servers else None

    response = sup.chat(
        user_query=text,
        summary=summary,
        recent_history=recent,
        allowed_servers=allowed,
    )

    answer = response["answer"]
    used = response.get("used_servers", [])
    needs_clarification = response.get("needs_clarification", False)

    db.save_message(chat_id, "assistant", answer,
                    meta={"used_servers": used, "needs_clarification": needs_clarification})

    if db.should_update_summary(chat_id):
        sup.update_summary_async(chat_id)

    return {"answer": answer, "used_servers": used, "needs_clarification": needs_clarification, "chat_id": chat_id}


def get_chat_history(chat_id: int, user_id: int) -> list[dict]:
    if not db.get_chat_safe(chat_id, user_id):
        return []
    return db.get_all_messages(chat_id)


def get_available_servers(user_id: int) -> list[str]:
    return get_supervisor(user_id).get_available_server_names()
