import os
from dotenv import load_dotenv

load_dotenv()

# LLM
API_KEY = os.getenv("API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

# SQLite
DATABASE_PATH = os.getenv("DATABASE_PATH", "assistant.db")

# Timeouts (секунды)
MCP_CONNECT_TIMEOUT = 10
MCP_REQUEST_TIMEOUT = 30
