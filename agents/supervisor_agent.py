"""
Supervisor Agent — главный координатор на основе LangGraph.

Граф: START → route → workers → synthesize → END
                    ↘ (no sources) → direct_llm → END

Шаги:
  1. route      — LLM анализирует запрос и выбирает релевантные серверы
  2. workers    — параллельный (или последовательный) запуск Worker Agents
  3. synthesize — LLM синтезирует финальный ответ из результатов workers
  4. direct_llm — если нет релевантных источников, отвечает напрямую
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
from database import get_servers, cache_tools, get_cached_tools
from config import API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# Максимум сообщений истории передаваемых в LLM:
MAX_HISTORY_MESSAGES = 0

class AssistantState(TypedDict):
    user_query: str
    # history: list[dict] # временно убрал историю из-за токенов, нужно потом реализовать новую логику для экономии токенов
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
        # Встроенный WebSearch всегда загружаем
        self._load_builtin_search()

    # Builtin search

    def _load_builtin_search(self):
        """Встроенный WebSearch доступен всегда."""
        tool = get_builtin_search_tool()
        worker = self.factory.create_from_lc_tools(BUILTIN_SERVER_NAME, [tool])
        self._workers[BUILTIN_SERVER_NAME] = worker
        logger.info("Built-in WebSearch loaded.")

    # Worker management

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

    def _build_history_messages(self, history: list[dict]) -> list:
        """
        Конвертирует историю чата в LangChain messages.
        Берём последние MAX_HISTORY_MESSAGES для экономии токенов.
        """
        messages = []
        recent = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    # Nodes

    def _node_route(self, state: AssistantState) -> dict:
        allowed = state.get("allowed_servers") or list(self._workers.keys())
        # Фильтруем только существующие
        allowed = [s for s in allowed if s in self._workers]

        if not allowed:
            return {"selected_servers": [], "worker_results": [], "use_direct_llm": True}

        summaries = self.get_worker_summaries(allowed)

        system = """You are a routing agent. Select which data sources are relevant for the user query.
Respond ONLY with a JSON array of source names.
Rules:
- Include WebSearch for current events, news, weather, facts, prices.
- Return [] for purely conversational messages (hi, thanks, jokes).
- Otherwise pick the most relevant sources."""

        user_msg = f"""Query: {state['user_query']}

Sources:
{summaries}

Choose from: {json.dumps(allowed)}
JSON array only:"""

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
        """Прямой ответ с историей чата."""
        messages = [SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Answer in the same language as the user. Be concise."
        ))]
        # Добавляем историю для контекста
        messages += self._build_history_messages(state.get("history", []))
        messages.append(HumanMessage(content=state["user_query"]))

        try:
            resp = self.llm.invoke(messages)
            return {"final_answer": resp.content, "used_servers": []}
        except Exception as e:
            return {"final_answer": f"Error: {e}", "used_servers": []}

    def _node_run_workers(self, state: AssistantState) -> dict:
        BUILTIN_MAP = {BUILTIN_SERVER_NAME: _ddg_search}
        results = []

        for server_name in state["selected_servers"]:
            logger.info(f"Running: {server_name}")

            if server_name in BUILTIN_MAP:
                result = BUILTIN_MAP[server_name](state["user_query"])
                logger.info(f"Built-in result: {result[:200]}")
            else:
                worker = self._workers.get(server_name)
                if not worker:
                    logger.warning(f"Worker not found: {server_name}")
                    result = f"⚠️ Server '{server_name}' not connected."
                else:
                    result = worker.run(state["user_query"])
                    logger.info(f"MCP result: {result[:200]}")

            results.append({"server": server_name, "result": result})

        return {"worker_results": results}

    def _node_synthesize(self, state: AssistantState) -> dict:
        worker_results = state.get("worker_results", [])

        if not worker_results:
            return self._node_direct_llm(state)

        results_text = ""
        used = []
        for item in worker_results:
            results_text += f"\n=== {item['server']} ===\n{item['result']}\n"
            used.append(item["server"])

        messages = [SystemMessage(content=(
            "You are a synthesis agent. Combine data from sources into a coherent answer. "
            "Answer in the same language as the user. Mention URLs when relevant."
        ))]
        # История для контекста (урезанная — экономим токены)
        messages += self._build_history_messages(state.get("history", []))[-4:]
        messages.append(HumanMessage(content=(
            f"Question: {state['user_query']}\n\n"
            f"Data:\n{results_text}\n\n"
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

    def chat(self, user_query: str,
             history: list[dict] = None,
             allowed_servers: list[str] = None) -> dict:
        """
        history         — последние сообщения чата из БД
        allowed_servers — серверы разрешённые в этом чате (None = все)
        """
        initial_state: AssistantState = {
            "user_query": user_query,
            "history": history or [],
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