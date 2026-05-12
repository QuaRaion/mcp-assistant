"""
Supervisor Agent — главный координатор на основе LangGraph.

Граф:
  START → route → clarify → END          (если запрос амбигвален)
               → workers → synthesize → END
               → direct_llm → END

Шаги:
  1. route      — LLM анализирует запрос + контекст, выбирает серверы или просит уточнить
  2. clarify    — возвращает уточняющий вопрос пользователю
  3. workers    — параллельный запуск Worker Agents через asyncio
  4. synthesize — LLM синтезирует финальный ответ
  5. direct_llm — если нет релевантных источников

Память:
  Summary memory = резюме старой истории + скользящее окно последних N сообщений.
  Резюме обновляется в фоновом потоке (threading.Thread).
"""

import json
import logging
import asyncio
import threading
import concurrent.futures
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

RECENT_MESSAGES_WINDOW = 6


class AssistantState(TypedDict):
    user_query: str
    summary: str
    recent_history: list[dict]
    allowed_servers: list[str]
    selected_servers: list[str]
    worker_results: Annotated[list[dict], operator.add]
    final_answer: str
    used_servers: list[str]
    use_direct_llm: bool
    needs_clarification: bool
    clarification_question: str


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

    # --- Memory helpers ---

    def _build_context_block(self, summary: str, recent_history: list[dict]) -> str:
        """Резюме + скользящее окно → единый текстовый блок для промптов."""
        parts = []
        if summary:
            parts.append(f"[Conversation summary]\n{summary}")
        if recent_history:
            lines = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            parts.append("[Recent messages]\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _build_lc_history(self, recent_history: list[dict]) -> list:
        result = []
        for msg in recent_history:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                result.append(AIMessage(content=msg["content"]))
        return result

    def update_summary(self, chat_id: int) -> str:
        """Пересчитывает резюме чата через LLM."""
        if not should_update_summary(chat_id):
            existing = get_summary(chat_id)
            return existing["summary"] if existing else ""

        all_msgs = get_messages(chat_id, limit=200)
        to_summarize = (
            all_msgs[:-RECENT_MESSAGES_WINDOW]
            if len(all_msgs) > RECENT_MESSAGES_WINDOW
            else []
        )
        if not to_summarize:
            return ""

        dialog_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in to_summarize
        ])

        old_summary_row = get_summary(chat_id)
        old_summary = old_summary_row["summary"] if old_summary_row else ""

        prompt_parts = []
        if old_summary:
            prompt_parts.append(f"Previous summary:\n{old_summary}\n")
        prompt_parts.append(
            f"New conversation fragment:\n{dialog_text}\n\n"
            "Write a concise summary (3-5 sentences) of the entire conversation. "
            "Include: main topics, key facts found (repos, tasks, pages), user context. "
            "Be specific. Reply with summary only."
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

    def update_summary_async(self, chat_id: int) -> None:
        """
        Запускает update_summary в фоновом daemon-потоке.
        Не блокирует UI — Streamlit продолжает работу сразу.
        """
        def _run():
            try:
                self.update_summary(chat_id)
            except Exception as e:
                logger.error(f"Background summary error for chat {chat_id}: {e}")

        thread = threading.Thread(target=_run, daemon=True, name=f"summary-{chat_id}")
        thread.start()
        logger.info(f"Summary background thread started for chat {chat_id}")

    # --- Async workers ---

    async def _run_worker_async(
        self,
        server_name: str,
        enriched_query: str,
        original_query: str,
    ) -> dict:
        """Запускает один воркер асинхронно через executor."""
        loop = asyncio.get_event_loop()

        if server_name == BUILTIN_SERVER_NAME:
            result = await loop.run_in_executor(None, _ddg_search, original_query)
        else:
            worker = self._workers.get(server_name)
            if not worker:
                logger.warning(f"Worker not found: {server_name}")
                result = f"⚠️ Server '{server_name}' not connected."
            else:
                result = await loop.run_in_executor(None, worker.run, enriched_query)

        logger.info(f"[{server_name}] completed: {str(result)[:150]}")
        return {"server": server_name, "result": result}

    async def _run_all_workers_async(
        self,
        selected_servers: list[str],
        enriched_query: str,
        original_query: str,
    ) -> list[dict]:
        """Параллельный запуск всех воркеров через asyncio.gather()."""
        tasks = [
            self._run_worker_async(name, enriched_query, original_query)
            for name in selected_servers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for name, res in zip(selected_servers, results):
            if isinstance(res, Exception):
                logger.error(f"Worker '{name}' raised exception: {res}")
                output.append({"server": name, "result": f"⚠️ Error: {res}"})
            else:
                output.append(res)
        return output

    def _run_workers_parallel(
        self,
        selected_servers: list[str],
        enriched_query: str,
        original_query: str,
    ) -> list[dict]:
        """
        Обёртка для вызова async кода из синхронного контекста LangGraph.
        Streamlit запускается в потоке где event loop может уже существовать,
        поэтому используем отдельный поток с собственным loop.
        """
        def _run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._run_all_workers_async(
                        selected_servers, enriched_query, original_query
                    )
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result()

    # --- Graph nodes ---

    def _node_route(self, state: AssistantState) -> dict:
        """
        Роутер с полным контекстом памяти.

        Промпт явно передаёт историю → модель понимает анафоры:
        "первого" = первый репо из предыдущего ответа.

        Три возможных исхода:
          route   → выбраны серверы
          direct  → прямой ответ без инструментов
          clarify → запрос слишком неоднозначен
        """
        allowed = state.get("allowed_servers") or list(self._workers.keys())
        allowed = [s for s in allowed if s in self._workers]

        if not allowed:
            return {
                "selected_servers": [], "worker_results": [],
                "use_direct_llm": True, "needs_clarification": False,
                "clarification_question": "",
            }

        summaries = self.get_worker_summaries(allowed)
        context_block = self._build_context_block(
            state.get("summary", ""),
            state.get("recent_history", []),
        )

        system = """You are a routing agent for an AI assistant. Analyze the user query and conversation context, then decide what to do.

RESPOND with JSON in exactly one of these formats:

1. Route to data sources (when query needs external data):
{"action": "route", "servers": ["Server1", "Server2"]}

2. Answer directly (greetings, simple questions, no tools needed):
{"action": "direct"}

3. Ask clarification (query is ambiguous AND context does not help resolve it):
{"action": "clarify", "question": "Your clarifying question in user language"}

IMPORTANT RULES:
- Always check conversation context FIRST to resolve references like "the first one", "that repo", "its issues", "покажи первый", "отправь второй"
- If context resolves the reference → route, do NOT clarify
- Use "clarify" only as last resort when truly impossible to determine intent
- Include WebSearch for: news, weather, current events, prices, general internet facts
- JSON only, no extra text"""

        context_part = f"\n\nConversation context:\n{context_block}" if context_block else ""
        user_msg = (
            f"User query: {state['user_query']}"
            f"{context_part}\n\n"
            f"Available sources:\n{summaries}\n\n"
            f"Valid server names: {json.dumps(allowed)}\n\n"
            f"JSON response:"
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_msg),
            ])
            raw = resp.content.strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError(f"No JSON in response: {raw}")

            parsed = json.loads(raw[start:end])
            action = parsed.get("action", "direct")

            if action == "route":
                servers = [s for s in parsed.get("servers", []) if s in self._workers]
                if not servers:
                    return {
                        "selected_servers": [], "worker_results": [],
                        "use_direct_llm": True, "needs_clarification": False,
                        "clarification_question": "",
                    }
                logger.info(f"Router → route: {servers}")
                return {
                    "selected_servers": servers, "worker_results": [],
                    "use_direct_llm": False, "needs_clarification": False,
                    "clarification_question": "",
                }

            elif action == "clarify":
                question = parsed.get("question", "Пожалуйста, уточните запрос.")
                logger.info(f"Router → clarify: {question}")
                return {
                    "selected_servers": [], "worker_results": [],
                    "use_direct_llm": False,
                    "needs_clarification": True,
                    "clarification_question": question,
                }

            else:  # direct
                logger.info("Router → direct")
                return {
                    "selected_servers": [], "worker_results": [],
                    "use_direct_llm": True, "needs_clarification": False,
                    "clarification_question": "",
                }

        except Exception as e:
            logger.error(f"Routing error: {e}")
            fallback = [BUILTIN_SERVER_NAME] if BUILTIN_SERVER_NAME in allowed else []
            return {
                "selected_servers": fallback, "worker_results": [],
                "use_direct_llm": not fallback, "needs_clarification": False,
                "clarification_question": "",
            }

    def _node_clarify(self, state: AssistantState) -> dict:
        """Возвращает уточняющий вопрос как финальный ответ."""
        question = state.get("clarification_question", "Пожалуйста, уточните ваш запрос.")
        return {"final_answer": question, "used_servers": []}

    def _node_direct_llm(self, state: AssistantState) -> dict:
        messages = [SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Answer in the same language as the user. Be concise."
        ))]
        summary = state.get("summary", "")
        if summary:
            messages.append(SystemMessage(content=f"Conversation context:\n{summary}"))
        messages += self._build_lc_history(state.get("recent_history", []))
        messages.append(HumanMessage(content=state["user_query"]))

        try:
            resp = self.llm.invoke(messages)
            return {"final_answer": resp.content, "used_servers": []}
        except Exception as e:
            return {"final_answer": f"Error: {e}", "used_servers": []}

    def _node_run_workers(self, state: AssistantState) -> dict:
        """Параллельный запуск воркеров. Enriched query содержит контекст памяти."""
        context_block = self._build_context_block(
            state.get("summary", ""),
            state.get("recent_history", []),
        )
        enriched_query = state["user_query"]
        if context_block:
            enriched_query = f"{context_block}\n\nCurrent request: {state['user_query']}"

        results = self._run_workers_parallel(
            state["selected_servers"],
            enriched_query,
            state["user_query"],
        )
        return {"worker_results": results}

    def _node_synthesize(self, state: AssistantState) -> dict:
        worker_results = state.get("worker_results", [])

        if not worker_results:
            return self._node_direct_llm(state)

        # Один источник с готовым развёрнутым ответом — пропускаем LLM
        if len(worker_results) == 1:
            single = worker_results[0]["result"]
            if isinstance(single, str) and len(single) > 100:
                logger.info("Single worker result — skipping synthesize LLM.")
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
            "You are a synthesis agent. Combine data from multiple sources into a coherent answer. "
            "Answer in the same language as the user. Mention URLs when relevant. Be concise."
        ))]

        summary = state.get("summary", "")
        if summary:
            messages.append(SystemMessage(content=f"Conversation context:\n{summary}"))

        messages += self._build_lc_history(state.get("recent_history", [])[-4:])
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

    # --- Graph wiring ---

    def _route_after_routing(self, state: AssistantState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        if state.get("use_direct_llm") or state.get("final_answer"):
            return "direct"
        return "workers" if state.get("selected_servers") else "direct"

    def _build_graph(self):
        graph = StateGraph(AssistantState)
        graph.add_node("route", self._node_route)
        graph.add_node("clarify", self._node_clarify)
        graph.add_node("direct_llm", self._node_direct_llm)
        graph.add_node("workers", self._node_run_workers)
        graph.add_node("synthesize", self._node_synthesize)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route", self._route_after_routing,
            {"clarify": "clarify", "direct": "direct_llm", "workers": "workers"},
        )
        graph.add_edge("workers", "synthesize")
        graph.add_edge("clarify", END)
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
            "needs_clarification": False,
            "clarification_question": "",
        }
        try:
            final_state = self._graph.invoke(initial_state)
            return {
                "answer": final_state.get("final_answer", "No answer."),
                "used_servers": final_state.get("used_servers", []),
                "needs_clarification": final_state.get("needs_clarification", False),
            }
        except Exception as e:
            logger.exception(f"Supervisor error: {e}")
            return {"answer": f"Error: {e}", "used_servers": [], "needs_clarification": False}
