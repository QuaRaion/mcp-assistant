"""
Встроенный Web Search агент на базе DuckDuckGo.
"""

import logging
from ddgs import DDGS
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

BUILTIN_SERVER_NAME = "WebSearch"


def _ddg_search(query: str, max_results: int = 3) -> str:
    """Поиск через DuckDuckGo, возвращает форматированный текст"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"{i}. **{title}**\n   {body}\n   URL: {href}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return f"Search error: {e}"


def get_builtin_search_tool() -> Tool:
    """Возвращает LangChain Tool для встроенного веб-поиска."""
    return Tool(
        name="web_search",
        description=(
            "Search the web for current information, news, facts, or any topic. "
            "Input: search query string. "
            "Use this when you need up-to-date information or facts you don't know."
        ),
        func=_ddg_search,
    )

BUILTIN_TOOLS_SCHEMA = [
    {
        "name": "web_search",
        "description": "Search the internet using DuckDuckGo. Returns top results with titles, snippets and URLs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]