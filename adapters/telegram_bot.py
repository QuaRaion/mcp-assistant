"""Telegram адаптер с приватными MCP серверами."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging, asyncio
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters, ContextTypes
from telegram.constants import ParseMode, ChatAction

import database as db
from mcp_client import MCPClient
from core.assistant import (
    process_message, create_new_chat, list_chats, get_chat, clear_chat,
    get_available_servers, reload_workers, invalidate_supervisor,
)
from config import TELEGRAM_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TG_MAX_LENGTH = 4000
ADD_NAME, ADD_URL, ADD_APIKEY = range(3)
REMOVE_PICK = 10  # отдельная константа чтобы не конфликтовала с ADD_*


def _split(text):
    if len(text) <= TG_MAX_LENGTH:
        return [text]
    parts = []
    while text:
        parts.append(text[:TG_MAX_LENGTH])
        text = text[TG_MAX_LENGTH:]
    return parts


def _esc(text):
    for ch in ["_", "*", "`", "["]:
        text = text.replace(ch, f"\\{ch}")
    return text


def _get_tg_user(update):
    tg = update.effective_user
    return db.get_or_create_telegram_user(tg.id, tg.username or "", tg.first_name or "")


def _get_active_chat(tg_user):
    user_id = tg_user["user_id"]
    active_id = tg_user.get("active_chat_id")
    if active_id:
        chat = get_chat(active_id, user_id)
        if chat:
            return chat
    chats = list_chats(user_id)
    chat = chats[0] if chats else create_new_chat(user_id, "Первый чат")
    db.set_telegram_active_chat(tg_user["telegram_id"], chat["id"])
    return chat


# ── Basic commands ────────────────────────────────────────────────────────────

async def cmd_start(update, context):
    tg_user = _get_tg_user(update)
    chat = _get_active_chat(tg_user)
    await update.message.reply_text(
        f"Привет, {tg_user.get('first_name') or 'там'}! 👋\n\n"
        f"Я MCP Assistant. Текущий чат: *{_esc(chat['title'])}*\n\n"
        f"/newchat /history /servers /addserver /help",
        parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update, context):
    await update.message.reply_text(
        "📋 *Команды:*\n\n"
        "*Чаты:* /newchat /history /switch N /clear\n"
        "*Серверы:* /servers /addserver /removeserver\n"
        "/help — справка", parse_mode=ParseMode.MARKDOWN)

async def cmd_newchat(update, context):
    tg_user = _get_tg_user(update)
    chat = create_new_chat(tg_user["user_id"], "Новый чат")
    db.set_telegram_active_chat(tg_user["telegram_id"], chat["id"])
    await update.message.reply_text("✅ Новый чат создан. Старая история сохранена — /history")

async def cmd_history(update, context):
    tg_user = _get_tg_user(update)
    chats = list_chats(tg_user["user_id"])
    if not chats:
        await update.message.reply_text("Нет чатов."); return
    active_id = tg_user.get("active_chat_id")
    lines = ["📚 *Ваши чаты:*\n"] + [
        f"{i}. {'▶ ' if c['id']==active_id else ''}*{_esc(c['title'])}*"
        for i, c in enumerate(chats[:5], 1)
    ] + ["\n/switch N — переключить"]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_switch(update, context):
    tg_user = _get_tg_user(update)
    chats = list_chats(tg_user["user_id"])
    try:
        n = int(context.args[0])
        assert 1 <= n <= len(chats)
    except Exception:
        await update.message.reply_text(f"Укажите число от 1 до {len(chats)}. Пример: /switch 2"); return
    chat = chats[n-1]
    db.set_telegram_active_chat(tg_user["telegram_id"], chat["id"])
    await update.message.reply_text(f"✅ Переключились на: *{_esc(chat['title'])}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_clear(update, context):
    tg_user = _get_tg_user(update)
    chat = _get_active_chat(tg_user)
    clear_chat(chat["id"], tg_user["user_id"])
    await update.message.reply_text(f"🗑️ История *{_esc(chat['title'])}* очищена.", parse_mode=ParseMode.MARKDOWN)

async def cmd_servers(update, context):
    tg_user = _get_tg_user(update)
    servers = db.get_servers(tg_user["user_id"], active_only=False)
    if not servers:
        await update.message.reply_text("Нет серверов.\n/addserver — добавить"); return
    lines = ["🔌 *Ваши MCP серверы:*\n"]
    for srv in servers:
        cached = db.get_cached_tools(srv["id"]) or []
        lines.append(f"{'🟢' if srv['is_active'] else '🔴'} *{_esc(srv['name'])}*\n"
                     f"   `{srv['url']}`\n   {len(cached)} инструментов")
    lines.append("\n/addserver — добавить\n/removeserver — удалить")
    await update.message.reply_text("\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ── /addserver conversation ───────────────────────────────────────────────────

async def addserver_start(update, context):
    context.user_data.clear()
    await update.message.reply_text(
        "🔌 *Добавление MCP сервера*\n\nШаг 1/3: Введите название\n_(GitHub, Notion...)_\n\n/cancel — отмена",
        parse_mode=ParseMode.MARKDOWN)
    return ADD_NAME

async def addserver_name(update, context):
    name = update.message.text.strip()
    if not name:
        await update.message.reply_text("Название не может быть пустым:"); return ADD_NAME
    tg_user = _get_tg_user(update)
    existing = db.get_servers(tg_user["user_id"], active_only=False)
    if any(s["name"].lower() == name.lower() for s in existing):
        await update.message.reply_text(f"❌ Сервер *{_esc(name)}* уже существует. Другое название:",
                                        parse_mode=ParseMode.MARKDOWN); return ADD_NAME
    context.user_data["name"] = name
    await update.message.reply_text(
        f"✅ Название: *{_esc(name)}*\n\nШаг 2/3: Введите URL\n_(http://localhost:8000/mcp)_\n\n/cancel",
        parse_mode=ParseMode.MARKDOWN)
    return ADD_URL

async def addserver_url(update, context):
    url = update.message.text.strip()
    if not url.startswith(("http://", "https://")):
        await update.message.reply_text("❌ URL должен начинаться с http:// или https://\nПовторите:"); return ADD_URL
    context.user_data["url"] = url
    await update.message.reply_text(
        f"✅ URL: `{url}`\n\nШаг 3/3: Введите API Key\n_(или `-` если не нужен)_\n\n/cancel",
        parse_mode=ParseMode.MARKDOWN)
    return ADD_APIKEY

async def addserver_apikey(update, context):
    raw = update.message.text.strip()
    api_key = "" if raw == "-" else raw
    name = context.user_data["name"]
    url = context.user_data["url"]
    tg_user = _get_tg_user(update)
    user_id = tg_user["user_id"]

    await update.message.reply_text(f"⏳ Проверяю *{_esc(name)}*...", parse_mode=ParseMode.MARKDOWN)

    loop = asyncio.get_event_loop()
    client = MCPClient(url, api_key)
    try:
        ok, message = await loop.run_in_executor(None, client.probe)
    except Exception as e:
        ok, message = False, str(e)

    if not ok:
        await update.message.reply_text(f"❌ Не удалось подключиться:\n`{_esc(message)}`\n\nПопробуйте /addserver снова.",
                                        parse_mode=ParseMode.MARKDOWN)
        context.user_data.clear(); return ConversationHandler.END

    try:
        server_id = db.add_server(user_id, name, url, api_key)
        tools = await loop.run_in_executor(None, client.get_tools_with_schema)
        if tools:
            db.cache_tools(server_id, user_id, tools)
        await loop.run_in_executor(None, lambda: reload_workers(user_id, force=True))

        tools_preview = "\n".join([f"  • `{t['name']}`" for t in tools[:8]])
        if len(tools) > 8:
            tools_preview += f"\n  _...и ещё {len(tools)-8}_"
        await update.message.reply_text(
            f"✅ *{_esc(name)}* подключён!\n\nИнструментов: {len(tools)}\n{tools_preview}",
            parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

    context.user_data.clear(); return ConversationHandler.END

async def addserver_cancel(update, context):
    context.user_data.clear()
    await update.message.reply_text("❌ Добавление отменено.")
    return ConversationHandler.END


# ── /removeserver conversation ────────────────────────────────────────────────

async def removeserver_start(update, context):
    tg_user = _get_tg_user(update)
    servers = [s for s in db.get_servers(tg_user["user_id"], active_only=False) if not s.get("is_builtin")]
    if not servers:
        await update.message.reply_text("Нет серверов для удаления."); return ConversationHandler.END
    lines = ["🗑️ *Какой сервер удалить?*\n"] + \
            [f"{i}. {'🟢' if s['is_active'] else '🔴'} {_esc(s['name'])}" for i, s in enumerate(servers, 1)] + \
            ["\nВведите номер или /cancel"]
    context.user_data["servers_to_remove"] = servers
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    return REMOVE_PICK

async def removeserver_pick(update, context):
    servers = context.user_data.get("servers_to_remove", [])
    try:
        n = int(update.message.text.strip())
        assert 1 <= n <= len(servers)
    except Exception:
        await update.message.reply_text(f"Введите число от 1 до {len(servers)} или /cancel"); return REMOVE_PICK
    srv = servers[n-1]
    tg_user = _get_tg_user(update)
    user_id = tg_user["user_id"]
    db.delete_server(srv["id"], user_id)
    loop = asyncio.get_event_loop()
    invalidate_supervisor(user_id)
    await loop.run_in_executor(None, lambda: reload_workers(user_id, force=True))
    await update.message.reply_text(f"✅ Сервер *{_esc(srv['name'])}* удалён.", parse_mode=ParseMode.MARKDOWN)
    context.user_data.clear(); return ConversationHandler.END

async def removeserver_cancel(update, context):
    context.user_data.clear()
    await update.message.reply_text("❌ Удаление отменено.")
    return ConversationHandler.END


# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update, context):
    tg_user = _get_tg_user(update)
    chat = _get_active_chat(tg_user)
    text = update.message.text.strip()
    await update.message.chat.send_action(ChatAction.TYPING)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_message, tg_user["user_id"], chat["id"], text)

    answer = result["answer"]
    if result.get("used_servers"):
        answer += f"\n\n📡 _{', '.join(result['used_servers'])}_"

    for part in _split(answer):
        try:
            await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(part)


# ── Bot setup ─────────────────────────────────────────────────────────────────

async def post_init(app):
    await app.bot.set_my_commands([
        BotCommand("newchat", "Новый чат"),
        BotCommand("history", "История чатов"),
        BotCommand("switch", "Переключить чат (/switch 2)"),
        BotCommand("clear", "Очистить чат"),
        BotCommand("servers", "Мои MCP серверы"),
        BotCommand("addserver", "Добавить MCP сервер"),
        BotCommand("removeserver", "Удалить MCP сервер"),
        BotCommand("help", "Справка"),
    ])


def run():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не задан в .env")
    db.init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    add_conv = ConversationHandler(
        entry_points=[CommandHandler("addserver", addserver_start)],
        states={
            ADD_NAME:   [MessageHandler(filters.TEXT & ~filters.COMMAND, addserver_name)],
            ADD_URL:    [MessageHandler(filters.TEXT & ~filters.COMMAND, addserver_url)],
            ADD_APIKEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, addserver_apikey)],
        },
        fallbacks=[CommandHandler("cancel", addserver_cancel)],
    )
    rm_conv = ConversationHandler(
        entry_points=[CommandHandler("removeserver", removeserver_start)],
        states={REMOVE_PICK: [MessageHandler(filters.TEXT & ~filters.COMMAND, removeserver_pick)]},
        fallbacks=[CommandHandler("cancel", removeserver_cancel)],
    )

    app.add_handler(add_conv)
    app.add_handler(rm_conv)
    for cmd, fn in [("start", cmd_start), ("help", cmd_help), ("newchat", cmd_newchat),
                    ("history", cmd_history), ("switch", cmd_switch),
                    ("clear", cmd_clear), ("servers", cmd_servers)]:
        app.add_handler(CommandHandler(cmd, fn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot started...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()
