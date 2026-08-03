import importlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "database" / "ecommerce.db"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

OPENAI_API_KEY = "***"
OLLAMA_API_KEY = "***"

# Set this to "gpt-5.4-mini", "gpt-4o-mini", "qwen2.5-7b_local", or "gemma4-31b".
LLM_MODEL = "gpt-4o-mini"

OPENAI_MODEL = "gpt-5.4-mini"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_GEMMA_BASE_URL = "https://ollama.com"
OLLAMA_GEMMA_MODEL = "gemma4:31b-cloud"

TEMPERATURE = 0.0

MAX_REASONING_STEPS = 15

SCENARIOS_PER_CONDITION = 5
API_CALL_DELAY = 1.0

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def build_llm(bind_tools=None):
    ChatOpenAI = importlib.import_module("langchain_openai").ChatOpenAI
    ChatOllama = importlib.import_module("langchain_ollama").ChatOllama

    if LLM_MODEL == "gpt-4o-mini":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
    elif LLM_MODEL == "qwen2.5-7b_local":
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
        )
    elif LLM_MODEL == "gemma4-31b":
        if OLLAMA_API_KEY:
            os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

        llm = ChatOllama(
            model=OLLAMA_GEMMA_MODEL,
            temperature=TEMPERATURE,
            base_url=OLLAMA_GEMMA_BASE_URL,
        )
    else:
        llm = ChatOpenAI(
            model=LLM_MODEL if LLM_MODEL != "" else OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )

    if bind_tools:
        return llm.bind_tools(bind_tools)
    return llm