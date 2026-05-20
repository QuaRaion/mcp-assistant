"""
Supervisor Agent — координатор на LangGraph.
load_workers() теперь принимает user_id и загружает только серверы этого пользователя.
"""

import json, logging, asyncio, threading, concurrent.futures, operator
from typing import TypedDict, Annotated
from agents.worker_agent import generate_server_hints

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from agents.worker_agent import WorkerAgent, WorkerAgentFactory, generate_server_hints
from agents.builtin_search import get_builtin_search_tool, BUILTIN_SERVER_NAME, _ddg_search
from mcp_client import MCPClient
from database import get_servers, cache_tools, get_cached_tools, get_summary, save_summary, should_update_summary, get_messages, count_messages, save_tools_hints, get_tools_hints
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
        self.llm = ChatOpenAI(model=LLM_MODEL, api_key=API_KEY, base_url=LLM_BASE_URL, temperature=0.7)
        self.factory = WorkerAgentFactory()
        self._workers: dict[str, WorkerAgent] = {}
        self._graph = self._build_graph()
        self._load_builtin_search()

    def _load_builtin_search(self):
        tool = get_builtin_search_tool()
        self._workers[BUILTIN_SERVER_NAME] = self.factory.create_from_lc_tools(BUILTIN_SERVER_NAME, [tool])

    def load_workers(self, user_id: int, force_refresh: bool = False) -> dict:
        """Загружает только серверы данного пользователя."""
        self._workers = {k: v for k, v in self._workers.items() if k == BUILTIN_SERVER_NAME}
        servers = get_servers(user_id=user_id, active_only=True)
        result = {BUILTIN_SERVER_NAME: ["web_search"]}

        for srv in servers:
            name = srv["name"]
            if force_refresh or not get_cached_tools(srv["id"]):
                try:
                    client = MCPClient(srv["url"], srv.get("api_key", ""))
                    client.initialize()
                    tools = client.list_tools()
                    cache_tools(srv["id"], user_id, tools)
                except Exception as e:
                    logger.error(f"Failed to load tools from {name}: {e}")
                    tools = []
            else:
                tools = get_cached_tools(srv["id"]) or []

            # После получения tools
            hints = get_tools_hints(srv["id"])
            if not hints and tools:  # генерируем один раз
                hints = generate_server_hints(name, tools, self.llm)
                save_tools_hints(srv["id"], user_id, hints)

            worker = self.factory.create(srv, tools, hints=hints)  # передаём hints

            worker = self.factory.create(srv, tools)
            if worker:
                self._workers[name] = worker
                result[name] = worker.tools_summary

        return result
    
    def get_available_server_names(self):
        return list(self._workers.keys())

    def get_worker_summaries(self, allowed=None):
        workers = self._workers
        if allowed:
            workers = {k: v for k, v in workers.items() if k in allowed}
        if not workers:
            return "No data sources available."
        return "\n".join([f"- {n}: [{', '.join(w.tools_summary) if w.tools_summary else 'no tools'}]"
                          for n, w in workers.items()])

    def _build_context_block(self, summary, recent_history):
        parts = []
        if summary:
            parts.append(f"[Conversation summary]\n{summary}")
        if recent_history:
            lines = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent_history]
            parts.append("[Recent messages]\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _build_lc_history(self, recent_history):
        result = []
        for msg in recent_history:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                result.append(AIMessage(content=msg["content"]))
        return result

    def update_summary(self, chat_id):
        if not should_update_summary(chat_id):
            existing = get_summary(chat_id)
            return existing["summary"] if existing else ""
        all_msgs = get_messages(chat_id, limit=200)
        to_summarize = all_msgs[:-RECENT_MESSAGES_WINDOW] if len(all_msgs) > RECENT_MESSAGES_WINDOW else []
        if not to_summarize:
            return ""
        dialog_text = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in to_summarize])
        old = get_summary(chat_id)
        old_summary = old["summary"] if old else ""
        prompt = (f"Previous summary:\n{old_summary}\n\n" if old_summary else "") + \
                 f"Conversation:\n{dialog_text}\n\nWrite 3-5 sentence summary. Key facts, topics, context. Summary only."
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            new_summary = resp.content.strip()
            save_summary(chat_id, new_summary, count_messages(chat_id))
            return new_summary
        except Exception as e:
            logger.error(f"Summary failed: {e}")
            return old_summary

    def update_summary_async(self, chat_id):
        threading.Thread(target=self.update_summary, args=(chat_id,), daemon=True).start()

    async def _run_worker_async(self, server_name, enriched_query, original_query):
        loop = asyncio.get_event_loop()
        if server_name == BUILTIN_SERVER_NAME:
            result = await loop.run_in_executor(None, _ddg_search, original_query)
        else:
            worker = self._workers.get(server_name)
            if not worker:
                result = f"⚠️ Server '{server_name}' not connected."
            else:
                result = await loop.run_in_executor(None, worker.run, enriched_query)
        return {"server": server_name, "result": result}

    async def _run_all_workers_async(self, selected, enriched, original):
        tasks = [self._run_worker_async(n, enriched, original) for n in selected]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = []
        for name, res in zip(selected, results):
            if isinstance(res, Exception):
                output.append({"server": name, "result": f"⚠️ Error: {res}"})
            else:
                output.append(res)
        return output

    def _run_workers_parallel(self, selected, enriched, original):
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run_all_workers_async(selected, enriched, original))
            finally:
                loop.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result()

    def _node_route(self, state):
        allowed = state.get("allowed_servers") or list(self._workers.keys())
        allowed = [s for s in allowed if s in self._workers]
        if not allowed:
            return {"selected_servers": [], "worker_results": [], "use_direct_llm": True,
                    "needs_clarification": False, "clarification_question": ""}

        context_block = self._build_context_block(state.get("summary",""), state.get("recent_history",[]))
        system = """You are a routing agent. Analyze the query and context, decide what to do.
Respond with JSON only:
{"action": "route", "servers": ["Name1"]}
{"action": "direct"}
{"action": "clarify", "question": "..."}
Rules: check context first to resolve references. Use clarify only as last resort. WebSearch for news/facts."""

        context_part = f"\n\nContext:\n{context_block}" if context_block else ""
        user_msg = f"Query: {state['user_query']}{context_part}\n\nSources:\n{self.get_worker_summaries(allowed)}\nValid names: {json.dumps(allowed)}\nJSON:"

        try:
            resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
            raw = resp.content.strip()
            s, e = raw.find("{"), raw.rfind("}") + 1
            parsed = json.loads(raw[s:e])
            action = parsed.get("action", "direct")

            if action == "route":
                servers = [s for s in parsed.get("servers", []) if s in self._workers]
                if not servers:
                    return {"selected_servers": [], "worker_results": [], "use_direct_llm": True,
                            "needs_clarification": False, "clarification_question": ""}
                return {"selected_servers": servers, "worker_results": [], "use_direct_llm": False,
                        "needs_clarification": False, "clarification_question": ""}
            elif action == "clarify":
                return {"selected_servers": [], "worker_results": [], "use_direct_llm": False,
                        "needs_clarification": True, "clarification_question": parsed.get("question", "Уточните запрос.")}
            else:
                return {"selected_servers": [], "worker_results": [], "use_direct_llm": True,
                        "needs_clarification": False, "clarification_question": ""}
        except Exception as e:
            logger.error(f"Routing error: {e}")
            fallback = [BUILTIN_SERVER_NAME] if BUILTIN_SERVER_NAME in allowed else []
            return {"selected_servers": fallback, "worker_results": [], "use_direct_llm": not fallback,
                    "needs_clarification": False, "clarification_question": ""}

    def _node_clarify(self, state):
        return {"final_answer": state.get("clarification_question", "Уточните запрос."), "used_servers": []}

    def _node_direct_llm(self, state):
        messages = [SystemMessage(content="You are a helpful AI assistant. Answer in the same language as the user. Be concise.")]
        summary = state.get("summary", "")
        if summary:
            messages.append(SystemMessage(content=f"Context:\n{summary}"))
        messages += self._build_lc_history(state.get("recent_history", []))
        messages.append(HumanMessage(content=state["user_query"]))
        try:
            return {"final_answer": self.llm.invoke(messages).content, "used_servers": []}
        except Exception as e:
            return {"final_answer": f"Error: {e}", "used_servers": []}

    def _node_run_workers(self, state):
        ctx = self._build_context_block(state.get("summary",""), state.get("recent_history",[]))
        enriched = f"{ctx}\n\nCurrent request: {state['user_query']}" if ctx else state["user_query"]
        return {"worker_results": self._run_workers_parallel(state["selected_servers"], enriched, state["user_query"])}

    def _node_synthesize(self, state):
        results = state.get("worker_results", [])
        if not results:
            return self._node_direct_llm(state)
        if len(results) == 1 and isinstance(results[0]["result"], str) and len(results[0]["result"]) > 100:
            return {"final_answer": results[0]["result"], "used_servers": [results[0]["server"]]}

        results_text = "".join([f"\n=== {r['server']} ===\n{r['result']}\n" for r in results])
        used = [r["server"] for r in results]
        messages = [SystemMessage(content="Synthesis agent. Combine sources into coherent answer. Same language as user. Be concise.")]
        summary = state.get("summary", "")
        if summary:
            messages.append(SystemMessage(content=f"Context:\n{summary}"))
        messages += self._build_lc_history(state.get("recent_history", [])[-4:])
        messages.append(HumanMessage(content=f"Question: {state['user_query']}\n\nData:\n{results_text}\n\nAnswer:"))
        try:
            return {"final_answer": self.llm.invoke(messages).content, "used_servers": used}
        except Exception as e:
            return {"final_answer": f"Error: {e}\n\n{results_text}", "used_servers": used}

    def _route_after_routing(self, state):
        if state.get("needs_clarification"):
            return "clarify"
        if state.get("use_direct_llm") or state.get("final_answer"):
            return "direct"
        return "workers" if state.get("selected_servers") else "direct"

    def _build_graph(self):
        g = StateGraph(AssistantState)
        g.add_node("route", self._node_route)
        g.add_node("clarify", self._node_clarify)
        g.add_node("direct_llm", self._node_direct_llm)
        g.add_node("workers", self._node_run_workers)
        g.add_node("synthesize", self._node_synthesize)
        g.add_edge(START, "route")
        g.add_conditional_edges("route", self._route_after_routing,
                                {"clarify": "clarify", "direct": "direct_llm", "workers": "workers"})
        g.add_edge("workers", "synthesize")
        g.add_edge("clarify", END)
        g.add_edge("direct_llm", END)
        g.add_edge("synthesize", END)
        return g.compile()

    def chat(self, user_query, summary="", recent_history=None, allowed_servers=None):
        state = {
            "user_query": user_query, "summary": summary, "recent_history": recent_history or [],
            "allowed_servers": allowed_servers or [], "selected_servers": [], "worker_results": [],
            "final_answer": "", "used_servers": [], "use_direct_llm": False,
            "needs_clarification": False, "clarification_question": "",
        }
        try:
            final = self._graph.invoke(state)
            return {"answer": final.get("final_answer","No answer."), "used_servers": final.get("used_servers",[]),
                    "needs_clarification": final.get("needs_clarification", False)}
        except Exception as e:
            logger.exception(f"Supervisor error: {e}")
            return {"answer": f"Error: {e}", "used_servers": [], "needs_clarification": False}
