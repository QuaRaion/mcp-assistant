"""
Universal MCP Client — подключение к MCP серверам через SSE транспорт.

Протокол MCP (Model Context Protocol):
  1. Клиент отправляет initialize запрос
  2. Сервер возвращает список capabilities
  3. Клиент вызывает tools/list для получения доступных инструментов
  4. Клиент вызывает tools/call для выполнения инструмента

Транспорт: SSE (Server-Sent Events) — наиболее распространённый.
"""

import json
import httpx
import logging
from typing import Any, Optional
from httpx_sse import connect_sse
from config import MCP_CONNECT_TIMEOUT, MCP_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class MCPError(Exception):
    pass


class MCPClient:
    """
    Универсальный SSE-клиент для MCP серверов.
    Поддерживает два стиля SSE MCP:
      - «новый» стиль: POST /mcp  (streamable HTTP, content-type text/event-stream)
      - «старый» стиль: GET /sse + POST /messages  (legacy SSE)
    """

    def __init__(self, url: str, api_key: str = ""):
        self.base_url = url.rstrip("/")
        self.api_key = api_key
        self._request_id = 0
        self._session_id = None

    # helpers

    # def _headers(self) -> dict:
    #     h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    #     if self.api_key:
    #         h["Authorization"] = f"Bearer {self.api_key}"
    #     return h

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
            h["Github-MCP-Installed-Apps"] = ""
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
    
    def _extract_session_id(self, resp: httpx.Response) -> None:
        """Извлекает и сохраняет session ID из заголовков ответа."""
        session_id = resp.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
            logger.info(f"Session ID saved: {session_id}")


    # low-level transport

    def _parse_response(self, resp: httpx.Response) -> dict:
        """Парсит ответ — JSON или SSE формат."""
        self._extract_session_id(resp)
        content_type = resp.headers.get("content-type", "")
        text = resp.text

        if "text/event-stream" in content_type or text.startswith("event:"):
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            return json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
            raise MCPError(f"Could not parse SSE response: {text[:200]}")

        return resp.json()

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=MCP_REQUEST_TIMEOUT) as client:
            resp = client.post(url, json=payload, headers=self._headers())
            logger.info(f"Request: {payload['method']}")
            logger.info(f"Response {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            return self._parse_response(resp)
        
    def _post_sse(self, endpoint: str, payload: dict) -> dict:
        """
        POST с ответом в виде SSE потока.
        Читаем события пока не получим финальный результат (event: message или data).
        """
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=MCP_REQUEST_TIMEOUT) as client:
            with connect_sse(client, "POST", url, json=payload, headers=self._headers()) as events:
                self._extract_session_id(events.response)
                for event in events.iter_sse():
                    if event.data and event.data != "[DONE]":
                        try:
                            data = json.loads(event.data)
                            if "result" in data or "error" in data:
                                return data
                        except json.JSONDecodeError:
                            continue
        raise MCPError("SSE stream ended without result")

    # MCP protocol

    def _call(self, method: str, params: Optional[dict] = None) -> Any:
        payload = self._rpc(method, params)
        endpoint = "/mcp"
        try:
            resp = self._post_json(endpoint, payload)
        except httpx.HTTPStatusError as e:
            raise MCPError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise MCPError(f"Cannot connect: {e}") from e

        if "error" in resp:
            raise MCPError(f"MCP error: {resp['error']}")
        return resp.get("result")

    def initialize(self) -> dict:
        """Инициализация сессии MCP (handshake)."""
        result = self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "mcp-assistant", "version": "1.0.0"},
        })
        return result or {}

    def list_tools(self) -> list[dict]:
        """Получить список доступных инструментов MCP сервера."""
        result = self._call("tools/list")
        if result is None:
            return []
        tools = result.get("tools", result) if isinstance(result, dict) else result
        return tools if isinstance(tools, list) else []

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Вызвать инструмент MCP сервера."""
        result = self._call("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # Результат может быть в разных форматах
        if result is None:
            return None
        if isinstance(result, dict):
            # Стандартный MCP: {content: [{type: "text", text: "..."}]}
            content = result.get("content", [])
            if content and isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif item.get("type") == "resource":
                            parts.append(str(item.get("resource", "")))
                return "\n".join(parts) if parts else str(result)
        return result

    # convenience

    def probe(self) -> tuple[bool, str]:
        """
        Проверить доступность сервера.
        Возвращает (ok: bool, message: str).
        """
        try:
            self.initialize()
            tools = self.list_tools()
            return True, f"Connected. Found {len(tools)} tool(s)."
        except MCPError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Connection error: {e}"

    def get_tools_with_schema(self) -> list[dict]:
        """
        Возвращает полные описания tools включая inputSchema.
        Формат, который используется для создания LangChain Tools.
        """
        try:
            self.initialize()
            return self.list_tools()
        except Exception as e:
            logger.error(f"Failed to list tools from {self.base_url}: {e}")
            return []
