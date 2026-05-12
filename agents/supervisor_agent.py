"""
Supervisor Agent — главный координатор на основе LangGraph.

Граф: START → route → workers → synthesize → END
                    ↘ (no sources) → direct_llm → END

Шаги:
  1. route      — LLM анализирует запрос и выбирает релевантные серверы
  2. workers    — запуск Worker Agents (последовательно)
  3. synthesize — LLM синтезирует финальный ответ из результатов workers
  4. direct_llm — если нет релевантных источников, отвечает напрямую

Память:
  Summary memory = резюме старой истории + скользящее окно последних N сообщений.
  Резюме хранится в SQLite, пересчитывается каждые SUMMARY_EVERY сообщений.
"""

import json
import logging
from typing import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from agents.worker_agent import WorkerAgent, WorkerAgentFactory
from agents.builtin_search import (
    get_builtin_search_tool,
    BUILTIN_SERVER_NAME,
    _ddg_search,
)
from mcp_client import MCPClient
from database import (
    get_servers, cache_tools, get_cached_tools,
    get_summary, save_summary, should_update_summary,
    get_messages, count_messages,
)
from config import API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# Скользящее окно: сколько последних сообщений передавать в LLM вместе с резюме.
# 6 = 3 пары user/assistant — достаточно для разрешения анафор ("первого", "его").
RECENT_MESSAGES_WINDOW = 6


class AssistantState(TypedDict):
    user_query: str
    summary: str            # резюме старой истории (может быть пустым)
    recent_history: list[dict]  # последние RECENT_MESSAGES_WINDOW сообщений
    allowed_servers: list[str]
    selected_servers: list[str]
    worker_results: Annotated[list[dict], operator.add]
    final_answer: str
    used_servers: list[str]
    use_direct_llm: bool


class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.7,
        )
        self.factory = WorkerAgentFactory()
        self._workers: dict[str, WorkerAgent] = {}
        self._graph = self._build_graph()
        self._load_builtin_search()

    # --- Builtin search ---

    def _load_builtin_search(self):
        tool = get_builtin_search_tool()
        worker = self.factory.create_from_lc_tools(BUILTIN_SERVER_NAME, [tool])
        self._workers[BUILTIN_SERVER_NAME] = worker
        logger.info("Built-in WebSearch loaded.")

    # --- Worker management ---

    def load_workers(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """Загружает MCP серверы из БД. Встроенный WebSearch остаётся всегда."""
        self._workers = {
            k: v for k, v in self._workers.items()
            if k == BUILTIN_SERVER_NAME
        }
        servers = get_servers(active_only=True)
        result = {BUILTIN_SERVER_NAME: ["web_search"]}

        for srv in servers:
            name = srv["name"]
            if force_refresh or not get_cached_tools(srv["id"]):
                try:
                    client = MCPClient(srv["url"], srv.get("api_key", ""))
                    client.initialize()
                    tools = client.list_tools()
                    cache_tools(srv["id"], tools)
                except Exception as e:
                    logger.error(f"Failed to load tools from {name}: {e}")
                    tools = []
            else:
                tools = get_cached_tools(srv["id"]) or []

            worker = self.factory.create(srv, tools)
            if worker:
                self._workers[name] = worker
                result[name] = worker.tools_summary

        return result

    def get_available_server_names(self) -> list[str]:
        return list(self._workers.keys())

    def get_worker_summaries(self, allowed: list[str] = None) -> str:
        workers = self._workers
        if allowed:
            workers = {k: v for k, v in workers.items() if k in allowed}
        if not workers:
            return "No data sources available."
        lines = []
        for name, w in workers.items():
            tools_str = ", ".join(w.tools_summary) if w.tools_summary else "no tools"
            lines.append(f"- {name}: [{tools_str}]")
        return "\n".join(lines)

    # --- Summary memory ---

    def _build_context_block(self, summary: str, recent_history: list[dict]) -> str:
        """
        Собирает блок контекста для передачи в LLM:
          [резюме прошлого] + [последние N сообщений].
        Пустые части пропускаются.
        """
        parts = []
        if summary:
            parts.append(f"[Conversation summary so far]\n{summary}")
        if recent_history:
            lines = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            parts.append("[Recent messages]\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _build_lc_history(self, recent_history: list[dict]) -> list:
        """Конвертирует recent_history в LangChain messages."""
        result = []
        for msg in recent_history:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                result.append(AIMessage(content=msg["content"]))
        return result

    def update_summary(self, chat_id: int) -> str:
        """
        Пересчитывает резюме чата через LLM.
        Вызывается из app.py после сохранения ответа ассистента,
        когда should_update_summary() возвращает True.
        Возвращает новое резюме (или старое если не нужно обновлять).
        """
        if not should_update_summary(chat_id):
            existing = get_summary(chat_id)
            return existing["summary"] if existing else ""

        # Берём все сообщения кроме последних RECENT_MESSAGES_WINDOW —
        # они уже будут в скользящем окне, резюмировать их не нужно.
        all_msgs = get_messages(chat_id, limit=200)
        to_summarize = all_msgs[:-RECENT_MESSAGES_WINDOW] if len(all_msgs) > RECENT_MESSAGES_WINDOW else []

        if not to_summarize:
            return ""

        # Формируем текст диалога для сжатия
        dialog_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in to_summarize
        ])

        # Если есть старое резюме — передаём его как контекст
        old_summary_row = get_summary(chat_id)
        old_summary = old_summary_row["summary"] if old_summary_row else ""

        prompt_parts = []
        if old_summary:
            prompt_parts.append(f"Previous summary:\n{old_summary}\n")
        prompt_parts.append(
            f"New conversation fragment:\n{dialog_text}\n\n"
            "Write a concise summary (3-5 sentences) of the entire conversation above. "
            "Include: main topics discussed, key facts found (repos, tasks, pages), "
            "user preferences or context. Be specific, not generic. "
            "Reply with summary only, no preamble."
        )

        try:
            resp = self.llm.invoke([HumanMessage(content="\n".join(prompt_parts))])
            new_summary = resp.content.strip()
            total = count_messages(chat_id)
            save_summary(chat_id, new_summary, total)
            logger.info(f"Summary updated for chat {chat_id}: {new_summary[:100]}")
            return new_summary
        except Exception as e:
            logger.error(f"Summary update failed: {e}")
            return old_summary

    # --- Graph nodes ---

    def _node_route(self, state: AssistantState) -> dict:
        allowed = state.get("allowed_servers") or list(self._workers.keys())
        allowed = [s for s in allowed if s in self._workers]

        if not allowed:
            return {"selected_servers": [], "worker_results": [], "use_direct_llm": True}

        summaries = self.get_worker_summaries(allowed)

        # Контекст для роутинга — нужен чтобы понять "первого" = repo1
        context_block = self._build_context_block(
            state.get("summary", ""),
            state.get("recent_history", []),
        )

        system = (
            "You are a routing agent. Select which data sources are relevant for the user query.\n"
            "Respond ONLY with a JSON array of source names.\n"
            "Rules:\n"
            "- Include WebSearch for current events, news, weather, facts.\n"
            "- Return [] for purely conversational messages (hi, thanks, jokes).\n"
            "- Consider conversation context when query references previous results."
        )

        context_part = f"\nContext:\n{context_block}\n" if context_block else ""
        user_msg = (
            f"Query: {state['user_query']}"
            f"{context_part}\n"
            f"Sources:\n{summaries}\n\n"
            f"Choose from: {json.dumps(allowed)}\n"
            f"JSON array only:"
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_msg),
            ])
            raw = resp.content.strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            if start != -1 and end > start:
                selected = json.loads(raw[start:end])
                selected = [s for s in selected if s in self._workers]
            else:
                selected = [BUILTIN_SERVER_NAME] if BUILTIN_SERVER_NAME in allowed else []
        except Exception as e:
            logger.error(f"Routing error: {e}")
            selected = [BUILTIN_SERVER_NAME] if BUILTIN_SERVER_NAME in allowed else []

        logger.info(f"Router selected: {selected}")

        if not selected:
            return {"selected_servers": [], "worker_results": [], "use_direct_llm": True}

        return {"selected_servers": selected, "worker_results": [], "use_direct_llm": False}

    def _node_direct_llm(self, state: AssistantState) -> dict:
        """Прямой ответ LLM с контекстом памяти."""
        messages = [SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Answer in the same language as the user. Be concise."
        ))]

        # Добавляем резюме как системный контекст
        summary = state.get("summary", "")
        if summary:
            messages.append(SystemMessage(content=f"Conversation context:\n{summary}"))

        # Скользящее окно истории
        messages += self._build_lc_history(state.get("recent_history", []))
        messages.append(HumanMessage(content=state["user_query"]))

        try:
            resp = self.llm.invoke(messages)
            return {"final_answer": resp.content, "used_servers": []}
        except Exception as e:
            return {"final_answer": f"Error: {e}", "used_servers": []}

    def _node_run_workers(self, state: AssistantState) -> dict:
        BUILTIN_MAP = {BUILTIN_SERVER_NAME: _ddg_search}
        results = []

        # Передаём контекст воркерам через расширенный запрос
        context_block = self._build_context_block(
            state.get("summary", ""),
            state.get("recent_history", []),
        )
        enriched_query = state["user_query"]
        if context_block:
            enriched_query = (
                f"{context_block}\n\n"
                f"Current request: {state['user_query']}"
            )

        for server_name in state["selected_servers"]:
            logger.info(f"Running worker: {server_name}")

            if server_name in BUILTIN_MAP:
                # WebSearch — передаём только оригинальный запрос
                result = BUILTIN_MAP[server_name](state["user_query"])
            else:
                worker = self._workers.get(server_name)
                if not worker:
                    logger.warning(f"Worker not found: {server_name}")
                    result = f"⚠️ Server '{server_name}' not connected."
                else:
                    # MCP воркер получает запрос с контекстом
                    result = worker.run(enriched_query)

            logger.info(f"[{server_name}] result preview: {str(result)[:200]}")
            results.append({"server": server_name, "result": result})

        return {"worker_results": results}

    def _node_synthesize(self, state: AssistantState) -> dict:
        worker_results = state.get("worker_results", [])

        if not worker_results:
            return self._node_direct_llm(state)

        # Если один источник и результат выглядит как готовый ответ — пропускаем LLM
        if len(worker_results) == 1:
            single = worker_results[0]["result"]
            # Если воркер уже дал развёрнутый ответ (>100 символов) — отдаём напрямую
            if isinstance(single, str) and len(single) > 100:
                logger.info("Single worker result — skipping synthesize LLM call.")
                return {
                    "final_answer": single,
                    "used_servers": [worker_results[0]["server"]],
                }

        results_text = ""
        used = []
        for item in worker_results:
            results_text += f"\n=== {item['server']} ===\n{item['result']}\n"
            used.append(item["server"])

        messages = [SystemMessage(content=(
            "You are a synthesis agent. Combine data from sources into a coherent answer. "
            "Answer in the same language as the user. Mention URLs when relevant. Be concise."
        ))]

        # Резюме как контекст
        summary = state.get("summary", "")
        if summary:
            messages.append(SystemMessage(content=f"Conversation context:\n{summary}"))

        # Последние 2 пары из скользящего окна (экономим токены)
        messages += self._build_lc_history(state.get("recent_history", [])[-4:])

        messages.append(HumanMessage(content=(
            f"Question: {state['user_query']}\n\n"
            f"Data from sources:\n{results_text}\n\n"
            f"Answer:"
        )))

        try:
            resp = self.llm.invoke(messages)
            return {"final_answer": resp.content, "used_servers": used}
        except Exception as e:
            return {"final_answer": f"Error: {e}\n\n{results_text}", "used_servers": used}

    def _route_after_routing(self, state: AssistantState) -> str:
        if state.get("use_direct_llm") or state.get("final_answer"):
            return "direct"
        return "workers" if state.get("selected_servers") else "direct"

    def _build_graph(self):
        graph = StateGraph(AssistantState)
        graph.add_node("route", self._node_route)
        graph.add_node("direct_llm", self._node_direct_llm)
        graph.add_node("workers", self._node_run_workers)
        graph.add_node("synthesize", self._node_synthesize)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route", self._route_after_routing,
            {"direct": "direct_llm", "workers": "workers"},
        )
        graph.add_edge("workers", "synthesize")
        graph.add_edge("direct_llm", END)
        graph.add_edge("synthesize", END)
        return graph.compile()

    def chat(
        self,
        user_query: str,
        summary: str = "",
        recent_history: list[dict] = None,
        allowed_servers: list[str] = None,
    ) -> dict:
        """
        summary        — резюме старой истории (из БД)
        recent_history — последние RECENT_MESSAGES_WINDOW сообщений (из БД)
        allowed_servers — серверы разрешённые в этом чате (None = все)
        """
        initial_state: AssistantState = {
            "user_query": user_query,
            "summary": summary,
            "recent_history": recent_history or [],
            "allowed_servers": allowed_servers or [],
            "selected_servers": [],
            "worker_results": [],
            "final_answer": "",
            "used_servers": [],
            "use_direct_llm": False,
        }
        try:
            final_state = self._graph.invoke(initial_state)
            return {
                "answer": final_state.get("final_answer", "No answer."),
                "used_servers": final_state.get("used_servers", []),
            }
        except Exception as e:
            logger.exception(f"Supervisor error: {e}")
            return {"answer": f"Error: {e}", "used_servers": []}
