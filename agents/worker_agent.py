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


class WorkerAgent:
    def __init__(self, server_name: str, server_id: int, llm, tools: list, tools_summary: list[str]):
        self.server_name = server_name
        self.server_id = server_id
        self.tools_summary = tools_summary
        self._llm = llm.bind(tool_choice="none") if hasattr(llm, "bind") else llm
        self._tools_map = {t.name: t for t in tools}

    def run(self, task: str) -> str:
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools_map.items()
        ])
        print(f">>> Tools available: {list(self._tools_map.keys())}")
        print(f">>> Tools descriptions:")
        for name, tool in self._tools_map.items():
            print(f"    {name}: {tool.description[:100]}")



        system = f"""You are working with '{self.server_name}' data source.
You have these tools available:
{tools_desc}

IMPORTANT: Do NOT use any built-in tool calling or JSON function calls.
You MUST respond using ONLY this exact text format:

To use a tool:
TOOL: tool_name
INPUT: tool_input

When done:
ANSWER: your final answer

Available tool names: {list(self._tools_map.keys())}
"""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=task),
        ]

        for _ in range(2):
            response = self._llm.invoke(messages)
            text = response.content.strip()

            if "ANSWER:" in text:
                return text.split("ANSWER:", 1)[1].strip()

            if "TOOL:" in text and "INPUT:" in text:
                try:
                    tool_line = [l for l in text.split("\n") if l.startswith("TOOL:")][0]
                    input_line = [l for l in text.split("\n") if l.startswith("INPUT:")][0]
                    tool_name = tool_line.replace("TOOL:", "").strip()
                    tool_input = input_line.replace("INPUT:", "").strip()

                    tool = self._tools_map.get(tool_name)
                    if not tool:
                        tool_result = f"Tool '{tool_name}' not found. Available: {list(self._tools_map.keys())}"
                    else:
                        tool_result = tool.func(tool_input)

                    messages.append(response)
                    messages.append(HumanMessage(content=f"Tool result:\n{tool_result}"))
                    continue
                except Exception as e:
                    messages.append(response)
                    messages.append(HumanMessage(content=f"Parsing error: {e}. Use exact format: TOOL: name\nINPUT: value"))
                    continue

            messages.append(response)
            messages.append(HumanMessage(content="Please use the exact format: TOOL: tool_name\nINPUT: value\nOr: ANSWER: your answer"))

        return "Could not get result after maximum iterations."


class WorkerAgentFactory:
    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )

    def create(self, server_info: dict, tools_schema: list[dict]) -> WorkerAgent | None:
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
        )

    def create_from_lc_tools(self, server_name: str, lc_tools: list[Tool]) -> WorkerAgent:
        return WorkerAgent(
            server_name=server_name,
            server_id=-1,
            llm=self._llm,
            tools=lc_tools,
            tools_summary=[t.name for t in lc_tools],
        )