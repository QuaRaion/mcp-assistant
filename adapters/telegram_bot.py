"""Telegram адаптер с inline-кнопками."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging, asyncio
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ConversationHandler,
    CallbackQueryHandler, filters, ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

import database as db
from mcp_client import MCPClient
from core.assistant import (
    process_message, create_new_chat, list_chats, get_chat,
    get_available_servers, reload_workers, invalidate_supervisor,
)
from config import TELEGRAM_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TG_MAX_LENGTH = 4000
ADD_NAME, ADD_URL, ADD_APIKEY = range(3)
REMOVE_PICK = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("➕ Новый чат", callback_data="newchat"),
            InlineKeyboardButton("📚 Мои чаты", callback_data="history"),
        ],
        [
            InlineKeyboardButton("🔌 MCP серверы", callback_data="servers"),
            InlineKeyboardButton("❓ Помощь", callback_data="help"),
        ],
    ])


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update, context):
    tg_user = _get_tg_user(update)
    chat = _get_active_chat(tg_user)
    name = tg_user.get("first_name") or "там"
    await update.message.reply_text(
        f"Привет, {name}\\! 👋\n\n"
        f"Я MCP Assistant — умный ассистент с доступом к вашим сервисам\\.\n\n"
        f"Активный чат: *{_esc(chat['title'])}*\n\n"
        f"Просто напишите вопрос или выберите действие:",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=_main_menu_keyboard(),
    )


async def cmd_menu(update, context):
    tg_user = _get_tg_user(update)
    chat = _get_active_chat(tg_user)
    await update.message.reply_text(
        f"Активный чат: *{_esc(chat['title'])}*\nВыберите действие:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_main_menu_keyboard(),
    )


# ── Callback handler ──────────────────────────────────────────────────────────

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    tg_user = _get_tg_user(update)

    # ── Новый чат
    if data == "newchat":
        chat = create_new_chat(tg_user["user_id"], "Новый чат")
        db.set_telegram_active_chat(tg_user["telegram_id"], chat["id"])
        await query.edit_message_text(
            "✅ Новый чат создан\\. Просто напишите сообщение\\!",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("‹ Меню", callback_data="menu")
            ]])
        )

    # ── История чатов
    elif data == "history":
        chats = list_chats(tg_user["user_id"])
        if not chats:
            await query.edit_message_text(
                "Чатов пока нет\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("‹ Меню", callback_data="menu")
                ]])
            )
            return
        active_id = tg_user.get("active_chat_id")
        buttons = []
        for c in chats[:8]:
            label = f"{'▶ ' if c['id'] == active_id else ''}{c['title']}"
            buttons.append([InlineKeyboardButton(label, callback_data=f"switch_{c['id']}")])
        buttons.append([InlineKeyboardButton("‹ Меню", callback_data="menu")])
        await query.edit_message_text(
            "📚 *Ваши чаты:*\nВыберите чат для переключения:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    # ── Переключение чата
    elif data.startswith("switch_"):
        chat_id = int(data.split("_")[1])
        chat = get_chat(chat_id, tg_user["user_id"])
        if chat:
            db.set_telegram_active_chat(tg_user["telegram_id"], chat_id)
            await query.edit_message_text(
                f"✅ Переключились на: *{_esc(chat['title'])}*",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🗑️ Удалить чат", callback_data=f"deletechat_{chat_id}"),
                    InlineKeyboardButton("‹ Меню", callback_data="menu"),
                ]])
            )

    # ── Удалить чат
    elif data.startswith("deletechat_"):
        chat_id = int(data.split("_")[1])
        user_id = tg_user["user_id"]
        chat = get_chat(chat_id, user_id)
        if chat:
            if tg_user.get("active_chat_id") == chat_id:
                db.set_telegram_active_chat(tg_user["telegram_id"], None)
            db.delete_chat(chat_id)
            await query.answer(f"Чат «{chat['title']}» удалён")
        chats = list_chats(user_id)
        if not chats:
            await query.edit_message_text(
                "Чатов больше нет\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("‹ Меню", callback_data="menu")
                ]])
            )
            return
        active_id = db.get_telegram_user(tg_user["telegram_id"]).get("active_chat_id")
        buttons = []
        for c in chats[:8]:
            label = f"{'▶ ' if c['id'] == active_id else ''}{c['title']}"
            buttons.append([InlineKeyboardButton(label, callback_data=f"switch_{c['id']}")])
        buttons.append([InlineKeyboardButton("‹ Меню", callback_data="menu")])
        await query.edit_message_text(
            "📚 *Ваши чаты:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    # ── MCP серверы
    elif data == "servers":
        servers = db.get_servers(tg_user["user_id"], active_only=False)
        lines = ["🔌 *MCP серверы:*\n"]
        for srv in servers:
            cached = db.get_cached_tools(srv["id"]) or []
            status = "🟢" if srv["is_active"] else "🔴"
            lines.append(f"{status} *{_esc(srv['name'])}* — {len(cached)} инструментов")
        if not servers:
            lines.append("_Нет подключённых серверов_")
        buttons = [
            [InlineKeyboardButton("➕ Добавить сервер", callback_data="addserver")],
        ]
        if servers:
            buttons.append([InlineKeyboardButton("🗑️ Удалить сервер", callback_data="removeserver")])
        buttons.append([InlineKeyboardButton("‹ Меню", callback_data="menu")])
        await query.edit_message_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    # ── Добавить сервер (запускаем ConversationHandler через сообщение)
    elif data == "addserver":
        await query.edit_message_text(
            "🔌 *Добавление MCP сервера*\n\n"
            "Шаг 1/3: Введите *название* сервера\n"
            "_Например: GitHub, Wiki, Plane_\n\n"
            "/cancel — отмена",
            parse_mode=ParseMode.MARKDOWN,
        )
        context.user_data["adding_server"] = True
        context.user_data["add_step"] = "name"

    # ── Удалить сервер
    elif data == "removeserver":
        servers = [s for s in db.get_servers(tg_user["user_id"], active_only=False)
                   if not s.get("is_builtin")]
        if not servers:
            await query.edit_message_text(
                "Нет серверов для удаления\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("‹ Назад", callback_data="servers")
                ]])
            )
            return
        buttons = []
        for srv in servers:
            label = f"{'🟢' if srv['is_active'] else '🔴'} {srv['name']}"
            buttons.append([InlineKeyboardButton(label, callback_data=f"del_{srv['id']}")])
        buttons.append([InlineKeyboardButton("‹ Назад", callback_data="servers")])
        await query.edit_message_text(
            "🗑️ *Выберите сервер для удаления:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    # ── Подтвердить удаление сервера
    elif data.startswith("del_"):
        server_id = int(data.split("_")[1])
        user_id = tg_user["user_id"]
        srv = db.get_server(server_id, user_id)
        if srv:
            db.delete_server(server_id, user_id)
            loop = asyncio.get_event_loop()
            invalidate_supervisor(user_id)
            await loop.run_in_executor(None, lambda: reload_workers(user_id, force=True))
            await query.edit_message_text(
                f"✅ Сервер *{_esc(srv['name'])}* удалён\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("‹ К серверам", callback_data="servers")
                ]])
            )

    # ── Помощь
    elif data == "help":
        await query.edit_message_text(
            "❓ *Как пользоваться:*\n\n"
            "Просто напишите вопрос в чат — ассистент сам выберет нужные источники\\.\n\n"
            "*Команды:*\n"
            "/menu — главное меню\n"
            "/newchat — новый чат\n",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("‹ Меню", callback_data="menu")
            ]])
        )

    # ── Вернуться в меню
    elif data == "menu":
        chat = _get_active_chat(tg_user)
        await query.edit_message_text(
            f"Активный чат: *{_esc(chat['title'])}*\nВыберите действие:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_main_menu_keyboard(),
        )


# ── Добавление сервера через текстовые шаги ───────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user = _get_tg_user(update)
    text = update.message.text.strip()

    # Шаги добавления сервера
    if context.user_data.get("adding_server"):
        step = context.user_data.get("add_step")

        if step == "name":
            if not text:
                await update.message.reply_text("Название не может быть пустым. Введите название:"); return
            existing = db.get_servers(tg_user["user_id"], active_only=False)
            if any(s["name"].lower() == text.lower() for s in existing):
                await update.message.reply_text(f"❌ Сервер *{_esc(text)}* уже существует. Введите другое название:",
                                                parse_mode=ParseMode.MARKDOWN); return
            context.user_data["srv_name"] = text
            context.user_data["add_step"] = "url"
            await update.message.reply_text(
                f"✅ Название: *{_esc(text)}*\n\nШаг 2/3: Введите *URL* сервера\n"
                f"_Например: http://localhost:8080_\n\n/cancel — отмена",
                parse_mode=ParseMode.MARKDOWN)
            return

        if step == "url":
            if not text.startswith(("http://", "https://")):
                await update.message.reply_text("❌ URL должен начинаться с http:// или https://\nПовторите:"); return
            context.user_data["srv_url"] = text
            context.user_data["add_step"] = "apikey"
            await update.message.reply_text(
                f"✅ URL: `{text}`\n\nШаг 3/3: Введите *API Key*\n"
                f"_Если не нужен — отправьте `-`_\n\n/cancel — отмена",
                parse_mode=ParseMode.MARKDOWN)
            return

        if step == "apikey":
            api_key = "" if text == "-" else text
            name = context.user_data["srv_name"]
            url = context.user_data["srv_url"]
            user_id = tg_user["user_id"]

            msg = await update.message.reply_text(f"⏳ Подключаюсь к *{_esc(name)}*...",
                                                  parse_mode=ParseMode.MARKDOWN)
            loop = asyncio.get_event_loop()
            client = MCPClient(url, api_key)
            try:
                ok, message = await loop.run_in_executor(None, client.probe)
            except Exception as e:
                ok, message = False, str(e)

            if not ok:
                context.user_data.clear()
                await msg.edit_text(f"❌ Не удалось подключиться:\n`{_esc(message)}`",
                                    parse_mode=ParseMode.MARKDOWN,
                                    reply_markup=InlineKeyboardMarkup([[
                                        InlineKeyboardButton("‹ К серверам", callback_data="servers")
                                    ]]))
                return

            try:
                server_id = db.add_server(user_id, name, url, api_key)
                tools = await loop.run_in_executor(None, client.get_tools_with_schema)
                if tools:
                    db.cache_tools(server_id, user_id, tools)
                await loop.run_in_executor(None, lambda: reload_workers(user_id, force=True))

                tools_preview = "\n".join([f"  • `{t['name']}`" for t in tools[:6]])
                if len(tools) > 6:
                    tools_preview += f"\n  _...и ещё {len(tools)-6}_"

                await msg.edit_text(
                    f"✅ *{_esc(name)}* подключён\\!\n\nИнструментов: {len(tools)}\n{tools_preview}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("‹ К серверам", callback_data="servers")
                    ]])
                )
            except Exception as e:
                await msg.edit_text(f"❌ Ошибка: {e}")

            context.user_data.clear()
            return

    # Обычное сообщение — обрабатываем как запрос
    chat = _get_active_chat(tg_user)
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


# ── Команды ───────────────────────────────────────────────────────────────────

async def cmd_newchat(update, context):
    tg_user = _get_tg_user(update)
    chat = create_new_chat(tg_user["user_id"], "Новый чат")
    db.set_telegram_active_chat(tg_user["telegram_id"], chat["id"])
    await update.message.reply_text("✅ Новый чат создан. Пишите вопрос!")

async def cmd_cancel(update, context):
    context.user_data.clear()
    await update.message.reply_text("❌ Отменено.", reply_markup=_main_menu_keyboard())


# ── Bot setup ─────────────────────────────────────────────────────────────────

async def post_init(app):
    await app.bot.set_my_commands([
        BotCommand("start", "Начать / главное меню"),
        BotCommand("menu", "Главное меню"),
        BotCommand("newchat", "Новый чат"),
    ])


def run():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не задан в .env")
    db.init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("newchat", cmd_newchat))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot started...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()