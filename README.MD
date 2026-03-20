### Структура проекта
```
mcp_assistant/
├── app.py                    # Streamlit UI (главный файл)
├── config.py                 # Настройки и конфигурация
├── database.py               # SQLite: хранение серверов и истории чата
├── mcp_client.py             # Universal MCP Client (SSE транспорт)
├── agents/
│   ├── __init__.py
│   ├── worker_agent.py       # Generic Worker Agent (ReAct + LangChain)
│   └── supervisor_agent.py   # Supervisor Agent (LangGraph)
├── requirements.txt
└── .env.example
```
где:
`database.py` — SQLite: MCP серверы, история чата
`mcp_client.py` — подключение к MCP серверам по SSE, получение tools
`worker_agent.py` — агент под конкретный MCP сервер, ReAct pattern
`supervisor_agent.py` — LangGraph граф: маршрутизация → workers → синтез
`app.py` — Streamlit чат + управление серверами (URL + API key)

### Компоненты и их функционал

**`config.py`** — читает `.env`, экспортирует константы (API ключ, модель, путь к БД, таймауты).

**`database.py`** — SQLite через стандартный `sqlite3`. Две таблицы: `mcp_servers` (url, api_key, кэш tools) и `chat_messages` (история). CRUD операции + кэширование tool-схем.

**`mcp_client.py`** — Universal MCP Client. Реализует JSON-RPC 2.0 поверх SSE/HTTP. Методы: `initialize()`(handshake), `list_tools()`, `call_tool()`, `probe()` (проверка связи). Автоматически парсит разные форматы ответа MCP серверов.

**`agents/worker_agent.py`** — `WorkerAgentFactory` создаёт `WorkerAgent` динамически: оборачивает каждый MCP tool в `LangChain Tool`, генерирует ReAct prompt с описанием сервера, создаёт `AgentExecutor`. Обрабатывает разные форматы входных данных от LLM (JSON / plain string).

**`agents/supervisor_agent.py`** — LangGraph `StateGraph` с тремя узлами: `route` (LLM выбирает релевантные серверы из списка), `workers` (запускает выбранных Worker Agents), `synthesize` (LLM объединяет результаты в связный ответ). Conditional edge пропускает workers если серверов нет.

**`app.py`** — Streamlit UI с тремя страницами:

- **Chat** — стандартный `st.chat_input`/`st.chat_message`, показывает какие серверы были использованы
- **Servers** — форма добавления (Name + URL + API Key), тест соединения, список с управлением (enable/disable/delete/refresh tools)
- **History** — полная история из БД с метаданными

---

### Инструкция по запуску
##### 1. Создать `.env` файл
```bash
cat > .env << EOF
API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
EOF
```

##### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

##### 3. Запустить
```bash
streamlit run app.py
```
