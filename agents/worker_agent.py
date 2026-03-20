"""
Generic Worker Agent — универсальный агент для работы с одним MCP сервером.

Паттерн: ReAct (Reasoning + Acting) через LangChain AgentExecutor.
Агент создаётся динамически на основе tools, полученных от MCP сервера.
"""

import json
import logging
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from mcp_client import MCPClient
from config import API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# ── ReAct prompt ───────────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template("""You are a specialized assistant working with the "{server_name}" data source.
Your job is to answer the user's question using ONLY the tools available from this source.

Available tools:
{tools}

Tool names: {tool_names}

Instructions:
- Use tools to retrieve real data, do not make up information.
- If a tool returns an error, report it honestly.
- Answer in the same language as the user's question.
- Be concise but complete.

Question: {input}

{agent_scratchpad}""")


# ── MCP Tool wrapper ───────────────────────────────────────────────────────

def _build_langchain_tool(mcp_client: MCPClient, tool_schema: dict) -> Tool:
    """
    Оборачивает MCP tool в LangChain Tool.
    tool_schema — описание из tools/list ответа MCP сервера.
    """
    name = tool_schema["name"]
    description = tool_schema.get("description", f"Tool: {name}")
    input_schema = tool_schema.get("inputSchema", {})

    # Добавляем схему в описание, чтобы LLM знал как передавать аргументы
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
        """Парсит входную строку и вызывает MCP tool."""
        try:
            # Пробуем распарсить как JSON
            if input_str.strip().startswith("{"):
                args = json.loads(input_str)
            else:
                # Если LLM передал просто строку — кладём в первый required параметр
                required_params = input_schema.get("required", [])
                if required_params:
                    args = {required_params[0]: input_str}
                else:
                    args = {"query": input_str}

            result = mcp_client.call_tool(name, args)
            return str(result) if result is not None else "No result returned."
        except json.JSONDecodeError:
            # Fallback: попытаться как простой запрос
            try:
                result = mcp_client.call_tool(name, {"query": input_str})
                return str(result) if result is not None else "No result returned."
            except Exception as e:
                return f"Error calling tool {name}: {e}"
        except Exception as e:
            return f"Error calling tool {name}: {e}"

    return Tool(name=name, description=full_description, func=tool_func)


# ── Worker Agent factory ───────────────────────────────────────────────────

class WorkerAgent:
    """
    Агент, специализированный на одном MCP сервере.
    Создаётся динамически через WorkerAgentFactory.
    """

    def __init__(self, server_name: str, server_id: int, executor: AgentExecutor,
                 tools_summary: list[str]):
        self.server_name = server_name
        self.server_id = server_id
        self.executor = executor
        self.tools_summary = tools_summary  # список названий tools

    def run(self, task: str) -> str:
        """Выполнить задачу. Возвращает строковый ответ."""
        try:
            result = self.executor.invoke({"input": task})
            return result.get("output", str(result))
        except Exception as e:
            logger.error(f"Worker [{self.server_name}] error: {e}")
            return f"[{self.server_name}] Failed to process task: {e}"


class WorkerAgentFactory:
    """Создаёт WorkerAgent для данного MCP сервера."""

    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )

    def create(self, server_info: dict, tools_schema: list[dict]) -> WorkerAgent | None:
        """
        server_info — запись из БД (id, name, url, api_key, ...)
        tools_schema — список tool описаний из MCP
        """
        if not tools_schema:
            logger.warning(f"No tools for server {server_info['name']}, skipping.")
            return None

        server_name = server_info["name"]
        mcp = MCPClient(server_info["url"], server_info.get("api_key", ""))

        lc_tools = [_build_langchain_tool(mcp, t) for t in tools_schema]

        prompt = REACT_PROMPT.partial(server_name=server_name)

        agent = create_react_agent(
            llm=self._llm,
            tools=lc_tools,
            prompt=prompt,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=lc_tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False,
        )

        return WorkerAgent(
            server_name=server_name,
            server_id=server_info["id"],
            executor=executor,
            tools_summary=[t["name"] for t in tools_schema],
        )

    def create_from_lc_tools(self, server_name: str, lc_tools: list[Tool]) -> WorkerAgent:
        """Создать WorkerAgent из уже готовых LangChain Tools (для встроенных агентов)."""
        prompt = REACT_PROMPT.partial(server_name=server_name)
        agent = create_react_agent(llm=self._llm, tools=lc_tools, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=lc_tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False,
        )
        return WorkerAgent(
            server_name=server_name,
            server_id=-1,  # builtin, нет в БД
            executor=executor,
            tools_summary=[t.name for t in lc_tools],
        )