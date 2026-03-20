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
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from agents.worker_agent import WorkerAgent, WorkerAgentFactory
from agents.builtin_search import (
    get_builtin_search_tool,
    BUILTIN_SERVER_NAME,
    BUILTIN_TOOLS_SCHEMA,
)
from mcp_client import MCPClient
from database import get_servers, cache_tools, get_cached_tools
from config import API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)


class AssistantState(TypedDict):
    user_query: str
    selected_servers: list[str]
    worker_results: Annotated[list[dict], operator.add]
    final_answer: str
    used_servers: list[str]
    use_direct_llm: bool  # флаг: отвечать напрямую без агентов


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
        # Сбрасываем всё кроме встроенных
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

    def get_worker_summaries(self) -> str:
        if not self._workers:
            return "No data sources available."
        lines = []
        for name, w in self._workers.items():
            tools_str = ", ".join(w.tools_summary) if w.tools_summary else "no tools"
            lines.append(f"- {name}: tools [{tools_str}]")
        return "\n".join(lines)

    def has_real_sources(self) -> bool:
        """Есть ли хоть один источник (включая встроенный WebSearch)."""
        return bool(self._workers)

    # LangGraph nodes

    def _node_route(self, state: AssistantState) -> dict:
        """Выбирает релевантные серверы или помечает direct_llm."""
        # Если вообще нет воркеров — прямой LLM
        if not self._workers:
            return {
                "selected_servers": [],
                "worker_results": [],
                "use_direct_llm": True,
            }

        summaries = self.get_worker_summaries()
        server_names = list(self._workers.keys())

        system = """You are a routing agent. Given a user query and available data sources, 
select which sources are relevant to answer the query.
Respond ONLY with a JSON array of server names from the list provided.
Important rules:
- Include web_search source for questions about current events, news, facts, or anything needing internet.
- Include web_search if the query might benefit from up-to-date information.
- If a user asks something purely conversational (hi, thanks, etc.), return [].
- Otherwise include all relevant sources.
Example: ["WebSearch", "Gmail"]"""

        user_msg = f"""User query: {state['user_query']}

Available sources:
{summaries}

Choose from: {json.dumps(server_names)}
Respond with JSON array only."""

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_msg),
            ])
            raw = resp.content.strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                selected = json.loads(raw[start:end])
                selected = [s for s in selected if s in self._workers]
            else:
                # Fallback: берём только WebSearch
                selected = [BUILTIN_SERVER_NAME] if BUILTIN_SERVER_NAME in self._workers else []
        except Exception as e:
            logger.error(f"Routing error: {e}")
            selected = [BUILTIN_SERVER_NAME] if BUILTIN_SERVER_NAME in self._workers else []

        logger.info(f"Router selected: {selected}")

        # Если роутер вернул пустой список — отвечаем напрямую (приветствие и т.п.)
        if not selected:
            return {
                "selected_servers": [],
                "worker_results": [],
                "use_direct_llm": True,
            }

        return {
            "selected_servers": selected,
            "worker_results": [],
            "use_direct_llm": False,
        }

    def _node_direct_llm(self, state: AssistantState) -> dict:
        """Прямой ответ LLM без использования источников данных."""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=(
                    "You are a helpful AI assistant. Answer the user's question clearly and helpfully. "
                    "Answer in the same language as the user's question."
                )),
                HumanMessage(content=state["user_query"]),
            ])
            return {"final_answer": resp.content, "used_servers": []}
        except Exception as e:
            return {"final_answer": f"Error: {e}", "used_servers": []}

    def _node_run_workers(self, state: AssistantState) -> dict:
        from agents.builtin_search import _ddg_search

        results = []

        BUILTIN_SERVERS = {
            BUILTIN_SERVER_NAME: _ddg_search,
        }

        for server_name in state["selected_servers"]:
            logger.info(f"Running: {server_name}")

            if server_name in BUILTIN_SERVERS:
                result = BUILTIN_SERVERS[server_name](state["user_query"])
                logger.info(f"Built-in result preview: {result[:200]}")
            else:
                worker = self._workers.get(server_name)
                if not worker:
                    logger.warning(f"MCP server not connected: {server_name}")
                    result = f"⚠️ Сервер '{server_name}' не подключен. Добавьте его в настройках."
                else:
                    result = worker.run(state["user_query"])
                    logger.info(f"MCP result preview: {result[:200]}")

            results.append({"server": server_name, "result": result})

        return {"worker_results": results}
    
    def _node_synthesize(self, state: AssistantState) -> dict:
        """Синтезирует финальный ответ из результатов workers."""
        worker_results = state.get("worker_results", [])

        if not worker_results:
            return (self._node_direct_llm(state))

        results_text = ""
        used = []
        for item in worker_results:
            srv = item["server"]
            res = item["result"]
            results_text += f"\n=== Data from {srv} ===\n{res}\n"
            used.append(srv)

        system = """You are a synthesis agent. Combine data from multiple sources into a coherent answer.
- Write a natural, well-structured response.
- Do not just copy-paste — synthesize the information.
- If sources conflict, note it.
- Answer in the same language as the user's question.
- For web search results, mention sources/URLs when relevant."""

        user_msg = f"""User question: {state['user_query']}

Collected data:
{results_text}

Provide a comprehensive answer."""

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_msg),
            ])
            return {"final_answer": resp.content, "used_servers": used}
        except Exception as e:
            return {"final_answer": f"Synthesis error: {e}\n\n{results_text}", "used_servers": used}

    # Routing logic

    def _route_after_routing(self, state: AssistantState) -> str:
        if state.get("use_direct_llm") or state.get("final_answer"):
            return "direct"
        if state.get("selected_servers"):
            return "workers"
        return "direct"

    # Graph

    def _build_graph(self):
        graph = StateGraph(AssistantState)

        graph.add_node("route", self._node_route)
        graph.add_node("direct_llm", self._node_direct_llm)
        graph.add_node("workers", self._node_run_workers)
        graph.add_node("synthesize", self._node_synthesize)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            self._route_after_routing,
            {"direct": "direct_llm", "workers": "workers"},
        )
        graph.add_edge("workers", "synthesize")
        graph.add_edge("direct_llm", END)
        graph.add_edge("synthesize", END)

        return graph.compile()

    # Public API

    def chat(self, user_query: str) -> dict:
        initial_state: AssistantState = {
            "user_query": user_query,
            "selected_servers": [],
            "worker_results": [],
            "final_answer": "",
            "used_servers": [],
            "use_direct_llm": False,
        }
        try:
            final_state = self._graph.invoke(initial_state)
            return {
                "answer": final_state.get("final_answer", "No answer generated."),
                "used_servers": final_state.get("used_servers", []),
            }
        except Exception as e:
            logger.exception(f"Supervisor error: {e}")
            return {"answer": f"An error occurred: {e}", "used_servers": []}
