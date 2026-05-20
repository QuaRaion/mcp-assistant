"""
Generic Worker Agent — универсальный агент для работы с одним MCP сервером.
Паттерн: текстовый ReAct loop совместимый с любыми LLM провайдерами.
"""

import json
import logging
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mcp_client import MCPClient
from config import API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# максимум итераций ReAct для обычных серверов
MAX_REACT_ITERATIONS = 20


def _build_langchain_tool(mcp_client: MCPClient, tool_schema: dict) -> Tool:
    name = tool_schema["name"]
    description = tool_schema.get("description", f"Tool: {name}")
    input_schema = tool_schema.get("inputSchema", {})

    schema_hint = ""
    if input_schema.get("properties"):
        props = input_schema["properties"]
        required = input_schema.get("required", [])
        params_desc = []
        for prop_name, prop_info in props.items():
            req_mark = " (required)" if prop_name in required else " (optional)"
            prop_type = prop_info.get("type", "string")
            prop_desc = prop_info.get("description", "")
            params_desc.append(f"  - {prop_name} [{prop_type}]{req_mark}: {prop_desc}")
        if params_desc:
            schema_hint = "\nParameters:\n" + "\n".join(params_desc)

    full_description = f"{description}{schema_hint}\nInput: JSON string with parameters."

    def tool_func(input_str: str) -> str:
        try:
            if not mcp_client._session_id:
                mcp_client.initialize()
            if input_str.strip().startswith("{"):
                args = json.loads(input_str)
            else:
                required_params = input_schema.get("required", [])
                if required_params:
                    args = {required_params[0]: input_str}
                else:
                    args = {"query": input_str}
            result = mcp_client.call_tool(name, args)
            return str(result) if result is not None else "No result returned."
        except json.JSONDecodeError:
            try:
                result = mcp_client.call_tool(name, {"query": input_str})
                return str(result) if result is not None else "No result returned."
            except Exception as e:
                return f"Error calling tool {name}: {e}"
        except Exception as e:
            return f"Error calling tool {name}: {e}"

    return Tool(name=name, description=full_description, func=tool_func)


def _build_system_prompt(server_name: str, tools_desc: str, tool_names: list[str], hints="") -> str:
    """
    Строит системный промпт для ReAct агента.

    Ключевые принципы для экономии токенов:
    - Жёсткий формат без отклонений
    - Пример прямо в промпте — модель сразу понимает что делать
    - Явный запрет на рассуждения вне формата
    - Правило «один инструмент за раз»
    """
    tools_json = json.dumps(tool_names)
    hints_block = f"\nSERVER-SPECIFIC RULES:\n{hints}" if hints else ""

    return f"""You are an agent working with '{server_name}'.

TOOLS:
{tools_desc}

STRICT FORMAT — use EXACTLY one of these two patterns per response:

Pattern 1 — call a tool:
TOOL: tool_name
INPUT: {{"param": "value"}}

Pattern 2 — give final answer:
ANSWER: your answer here

RULES:
- One pattern per response, nothing else.
- NEVER add explanations before TOOL/ANSWER.
- NEVER call a tool you already called with the same input.
- If tool result contains the answer — use ANSWER immediately.
- If task needs no tools — use ANSWER immediately.
- Available tools: {tools_json}
- Answer in the same language as the user.

GITHUB SPECIFICS (if applicable):
- Need username? → TOOL: get_me / INPUT: {{}}  (use login from result for next calls)
- List repos: TOOL: search_repositories / INPUT: {{"query": "user:LOGIN"}}
- List issues: TOOL: list_issues / INPUT: {{"owner": "LOGIN", "repo": "REPONAME"}}

EXAMPLE:
User: list my repos
Response:
TOOL: get_me
INPUT: {{}}
[tool returns: {{"login": "alice", ...}}]
TOOL: search_repositories
INPUT: {{"query": "user:alice"}}
[tool returns list]
ANSWER: Your repositories: repo1, repo2, repo3
{hints_block}
"""

def generate_server_hints(server_name: str, tools_schema: list[dict], llm) -> str:
    tools_desc = "\n".join([
        f"- {t['name']}: {t.get('description', '')}"
        for t in tools_schema
    ])
    prompt = f"""You are analyzing tools of MCP server '{server_name}'.

Tools:
{tools_desc}

Identify if any tools must be called in a specific sequence or have dependencies.
For example: 'to list user repos you must first call get_me to get the username'.

Write 3-5 short rules in this format:
- To do X: first call tool_a, then use result in tool_b.

If there are no dependencies, write: No special sequences required.
Be concise. Rules only, no explanations."""

    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception:
        return ""

class WorkerAgent:
    def __init__(self, server_name: str, server_id: int, llm, tools: list, tools_summary: list[str], hints=""):
        self.server_name = server_name
        self.server_id = server_id
        self.tools_summary = tools_summary
        self._llm = llm
        self._tools_map = {t.name: t for t in tools}
        self._cache = {}
        self.hints = hints

    def run(self, task: str) -> str:
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools_map.items()
        ])

        system = _build_system_prompt(
            self.server_name,
            tools_desc,
            list(self._tools_map.keys()),
            hints=self.hints,
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=task),
        ]

        called_tools: set[str] = set()  # защита от повторных вызовов

        for iteration in range(MAX_REACT_ITERATIONS):
            response = self._llm.invoke(messages)
            text = response.content.strip()
            logger.info(f"[{self.server_name}] iter={iteration} response={text[:120]}")

            # Финальный ответ
            if "ANSWER:" in text:
                return text.split("ANSWER:", 1)[1].strip()

            # Вызов инструмента
            if "TOOL:" in text and "INPUT:" in text:
                try:
                    lines = text.split("\n")
                    tool_name = next(
                        l.replace("TOOL:", "").strip() for l in lines if l.startswith("TOOL:")
                    )
                    tool_input = next(
                        l.replace("INPUT:", "").strip() for l in lines if l.startswith("INPUT:")
                    )

                    tool = self._tools_map.get(tool_name)
                    if not tool:
                        tool_result = (
                            f"Tool '{tool_name}' not found. "
                            f"Available: {list(self._tools_map.keys())}"
                        )
                    else:
                        # Защита от дублирующих вызовов
                        call_key = f"{tool_name}:{tool_input}"
                        if call_key in called_tools:
                            tool_result = "You already called this tool with the same input. Use ANSWER now."
                        elif call_key in self._cache:
                            tool_result = self._cache[call_key]
                            logger.info(f"Cache hit: {tool_name}")
                        else:
                            tool_result = tool.func(tool_input)
                            called_tools.add(call_key)
                            # Кэшируем read-only вызовы
                            CACHEABLE = {
                                "get_me", "list_repositories", "search_repositories",
                                "list_branches", "get_file_contents", "list_issues",
                                "list_pull_requests", "get_commit",
                            }
                            if tool_name in CACHEABLE:
                                self._cache[call_key] = tool_result
                                logger.info(f"Cached: {tool_name}")

                    # большие результаты обрезаются:
                    if len(str(tool_result)) > 10000:
                        tool_result = str(tool_result)[:10000] + "... [truncated]"

                    messages.append(response)
                    messages.append(HumanMessage(content=f"Tool result:\n{tool_result}"))
                    continue

                except (StopIteration, Exception) as e:
                    messages.append(response)
                    messages.append(HumanMessage(
                        content=f"Parse error: {e}. Use EXACTLY:\nTOOL: name\nINPUT: {{\"key\": \"value\"}}"
                    ))
                    continue

            # Модель не соблюла формат — жёсткое напоминание
            messages.append(response)
            messages.append(HumanMessage(
                content="FORMAT ERROR. Respond with ONLY:\nTOOL: tool_name\nINPUT: {\"key\": \"value\"}\nOR:\nANSWER: your answer"
            ))

        return "Could not complete the task within iteration limit."


class WorkerAgentFactory:
    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
            model_kwargs={"tool_choice": "none"},
        )

    def create(self, server_info: dict, tools_schema: list[dict], hints="") -> WorkerAgent | None:
        if not tools_schema:
            logger.warning(f"No tools for server {server_info['name']}, skipping.")
            return None

        server_name = server_info["name"]
        mcp = MCPClient(server_info["url"], server_info.get("api_key", ""))
        lc_tools = [_build_langchain_tool(mcp, t) for t in tools_schema]

        return WorkerAgent(
            server_name=server_name,
            server_id=server_info["id"],
            llm=self._llm,
            tools=lc_tools,
            tools_summary=[t["name"] for t in tools_schema],
            hints=hints,
        )

    def create_from_lc_tools(self, server_name: str, lc_tools: list[Tool]) -> WorkerAgent:
        return WorkerAgent(
            server_name=server_name,
            server_id=-1,
            llm=self._llm,
            tools=lc_tools,
            tools_summary=[t.name for t in lc_tools],
        )
