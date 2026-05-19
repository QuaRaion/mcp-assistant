# MCP Assistant

**Мультиагентный интеллектуальный ассистент** на базе LLM и [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

Мультиагентная система на базе LLM и MCP с plug-and-play архитектурой подключения внешних сервисов. 
Агрегирует ответы из GitHub, Plane, Wiki.js, веб-поиска и любых других MCP-совместимых источников. 
В качестве демонстрационных адаптеров реализованы Streamlit web UI и Telegram-бот.

---

## Содержание

- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [Требования](#требования)
- [Установка](#установка)
- [Настройка](#настройка)
- [Запуск](#запуск)
  - [MCP серверы](#запуск-mcp-серверов)
  - [Streamlit UI](#запуск-streamlit-ui)
  - [Telegram бот](#запуск-telegram-бота)
- [Подключение MCP серверов](#подключение-mcp-серверов)
- [Структура проекта](#структура-проекта)
- [Технический стек](#технический-стек)

---

## Возможности

- Запросы принимаются на естественном языке - система сама определяет к каким источникам обратиться.
- Supervisor Agent анализирует запрос и автоматически выбирает релевантные MCP-серверы, при необходимости запрашивая уточнение.
- Worker Agents обращаются к источникам параллельно, что сокращает время ответа при работе с несколькими серверами одновременно.
- Любой MCP-совместимый сервер подключается через URL и API-ключ прямо из интерфейса без изменения кода и перезапуска приложения.
- Веб-поиск через DuckDuckGo встроен и всегда доступен без дополнительной настройки и API-ключей.
- Ассистент помнит контекст разговора: каждые 10 сообщений история сворачивается в LLM-резюме, а последние 6 сообщений всегда передаются в полном виде.
- У каждого пользователя свои чаты, свои подключённые серверы и изолированная история — данные не пересекаются между аккаунтами.
- Платформонезависимое ядро (`core/assistant.py`) изолировано от интерфейсов через слой адаптеров — Streamlit и Telegram-бот реализованы как демонстрационные адаптеры, к ядру можно подключить любой другой интерфейс без изменения логики.
- Для демонстрации используется упрощенная БД SQLite. При необходимости переход на PostgreSQL или другую реляционную БД требует замены одной строки конфигурации.
---

## Архитектура

```
Запрос пользователя
                │
                ▼
        ┌─────────────────--┐
        │ Supervisor  Agent │  ← LangGraph граф
        │  (маршрутизация)  │
        └───────┬─────────--┘
                │
    ┌───────────┴──────────┐
    │                      │
    ▼                      ▼
┌────────────┐      ┌────────────┐
│Worker Agent│      │Worker Agent│   (параллельно)
│  GitHub    │      │  Wiki.js   │
└─────┬──────┘      └─────┬──────┘
      │                   │
      └─────────┬─────────┘
                ▼
        ┌───────────────┐
        │  Synthesize   │  ← объединение ответов
        └───────────────┘
                │
                ▼
         Ответ пользователю
```

Граф состояний: `START → route → workers / direct_llm / clarify → synthesize → END`

Вместо нативного tool calling используется текстовый **ReAct loop** (формат `TOOL / INPUT / ANSWER`) — совместим с любыми LLM провайдерами.

---

## Требования

- Python 3.12
- Node.js 20+ (для MCP серверов Wiki.js и Plane)
- `uv` или `pip`

---

## Установка

```bash
git clone https://github.com/QuaRaion/mcp-assistant
cd mcp-assistant
git checkout feature/multi-frontend

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Настройка

Создайте файл `.env` в корне проекта:

```env
# LLM провайдер (DeepSeek)
API_KEY=sk-ваш_ключ
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# База данных
DATABASE_PATH=data/assistant.db

# Telegram (опционально)
TELEGRAM_TOKEN=ваш_токен_бота
```

Создайте папку для базы данных:

```bash
mkdir -p data
```

### Получение API ключей

| Сервис | Где получить |
|--------|-------------|
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) |
| Telegram Bot | [@BotFather](https://t.me/BotFather) → `/newbot` |
| GitHub MCP | [github.com/settings/tokens](https://github.com/settings/tokens) |
| Wiki.js MCP | Administration → API Access → Generate Token |
| Plane MCP | Settings → API Tokens → Create Token |

---

## Запуск

### Запуск MCP серверов

MCP серверы запускаются отдельно перед основным приложением.

**Wiki.js** (требует запущенного контейнера Wiki.js):

```bash
# Сначала запустите Wiki.js если не запущен
docker run -d --name wikijs -p 3000:3000 \
  -e DB_TYPE=sqlite -e DB_FILEPATH=/wiki/db.sqlite \
  requarks/wiki:2

# Затем MCP сервер
WIKIJS_URL=http://localhost:3000 \
WIKIJS_API_KEY=ваш_api_ключ \
TRANSPORT_METHOD=streamable-http \
TRANSPORT_OPTIONS_PORT=8080 \
npx wikijs-mcp

# Слушает: http://localhost:8080/mcp
```

**Plane** (требует запущенного Plane и прокси-сервера из папки `plane-mcp-proxy/`):

```bash
# Сначала запустите Plane (из папки с plane.env и docker-compose.yml)
docker compose --env-file plane.env up -d

# Затем HTTP прокси для MCP
cd plane-mcp-proxy

PLANE_API_KEY=ваш_api_ключ \
PLANE_WORKSPACE_SLUG=ваш_slug \
PLANE_API_HOST_URL=http://localhost \
PORT=8211 \
node server.js

# Слушает: http://localhost:8211/mcp
```

**GitHub MCP** — внешний сервер, не требует локального запуска. Подключается напрямую через UI по адресу `https://api.githubcopilot.com` с OAuth токеном.

### Запуск Streamlit UI

```bash
streamlit run adapters/streamlit_app.py
```

Откроется браузер на `http://localhost:8501`. Зарегистрируйте аккаунт и войдите.

### Запуск Telegram бота

```bash
python adapters/telegram_bot.py
```

Доступные команды в боте:

| Команда | Описание |
|---------|----------|
| `/start` | Начало работы |
| `/newchat` | Создать новый чат |
| `/history` | Список чатов |
| `/switch N` | Переключиться на чат N |
| `/clear` | Очистить историю чата |
| `/servers` | Список MCP серверов |
| `/addserver` | Добавить MCP сервер |
| `/removeserver` | Удалить MCP сервер |
| `/help` | Справка |

---

## Подключение MCP серверов

После входа в систему перейдите в раздел **🔌 MCP серверы** и нажмите **Добавить**.

> **Важно:** вводите базовый URL **без** `/mcp` в конце — клиент добавляет суффикс автоматически.

| Сервер | URL | API Key |
|--------|-----|---------|
| Wiki.js MCP | `http://localhost:8080` | ключ из Wiki.js |
| Plane MCP | `http://localhost:8211` | не нужен |
| GitHub MCP | `https://api.githubcopilot.com` | OAuth токен |

После подключения система автоматически загружает список инструментов сервера и кэширует их. При последующих запросах маршрутизатор будет учитывать эти инструменты при выборе источников.

---

## Структура проекта

```
mcp_assistant/
├── core/
│   └── assistant.py          # ядро: process_message(), кэш SupervisorAgent per-user
├── agents/
│   ├── supervisor_agent.py   # LangGraph граф, параллельные воркеры, summary memory
│   ├── worker_agent.py       # текстовый ReAct loop, MAX_REACT_ITERATIONS=3
│   └── builtin_search.py     # встроенный DuckDuckGo поиск
├── adapters/
│   ├── streamlit_app.py      # Streamlit UI с авторизацией
│   └── telegram_bot.py       # Telegram бот (ConversationHandler)
├── plane-mcp-proxy/
│   ├── server.js             # HTTP→stdio прокси для Plane MCP
│   └── Dockerfile            # образ прокси (для будущего Docker запуска)
├── database.py               # SQLite: 6 таблиц, изоляция по user_id
├── mcp_client.py             # HTTP/SSE транспорт MCP, поддержка Mcp-Session-Id
├── config.py                 # конфигурация из .env
├── requirements.txt
├── .env                      # секреты (не в git)
└── data/
    └── assistant.db          # SQLite база данных (не в git)
```

---

## Технический стек

| Компонент | Технология |
|-----------|------------|
| LLM провайдер | DeepSeek API (OpenAI-совместимый) |
| Оркестрация | LangChain + LangGraph |
| Web UI | Streamlit |
| Telegram | python-telegram-bot 21.9 |
| MCP транспорт | SSE / Streamable HTTP (httpx + httpx-sse) |
| База данных | SQLite |
| Веб-поиск | DuckDuckGo (ddgs) |
| Python | 3.12 |
