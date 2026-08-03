"""
Central Configuration for the Multi-Agent System
The entire set of configuration parameters for runtime behavior is
defined in this single location so that experimentation can be made repeatable
and modifications to parameters do not get mixed up with application logic.
Any module requiring a particular parameter will import it from this module
instead of hard coding the value into each module individually.

This version supports multiple LLM backends (OpenAI and Ollama) for
evaluating different models across the research hypotheses.
"""

import importlib
import os
from pathlib import Path

# PROJECT_ROOT resolves to the `hybrid/` directory
PROJECT_ROOT = Path(__file__).parent

# Path to the SQLite database file.  The database is recreated from scratch
# before each evaluation run to guarantee a clean, reproducible initial state.
DATABASE_PATH = PROJECT_ROOT / "database" / "ecommerce.db"

# Directory where evaluation results (JSON files and log files) are written.
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Project-scoped API key used exclusively for thesis experiments.
OPENAI_API_KEY = "***"

# Ollama API key for remote model access (Gemma4-31B cloud).
OLLAMA_API_KEY = "***"

# Active LLM backend selector.  Set this to one of:
#   "gpt-4o-mini"        -- OpenAI GPT-4o-mini (cloud)
#   "gpt-5.4-mini"       -- OpenAI GPT-5.4-mini (cloud, via else branch)
#   "qwen2.5-7b_local"   -- Qwen 2.5 7B via local Ollama server
#   "gemma4-31b"         -- Gemma4 31B via remote Ollama cloud
LLM_MODEL = "gpt-4o-mini"

# The default OpenAI model used when LLM_MODEL is not recognised.
OPENAI_MODEL = "gpt-5.4-mini"

# Ollama connection settings for the local Qwen2.5-7B model.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Ollama connection settings for the remote Gemma4-31B cloud model.
OLLAMA_GEMMA_BASE_URL = "https://ollama.com"
OLLAMA_GEMMA_MODEL = "gemma4:31b-cloud"

# Temperature controls LLM output randomness.  0.0 = fully deterministic,
# which is essential for reproducible experiments (Hypothesis H2 in particular
# measures consistency, so stochastic outputs would introduce noise).
TEMPERATURE = 0.0

# Hard upper bound on the number of graph iterations (node executions) per
# workflow run.  If this limit is reached the workflow is escalated to a human
# agent rather than looping indefinitely.  This guards against pathological
# LLM-generated task plans that expand without bound.
MAX_REASONING_STEPS = 15

# Number of independent runs executed for each (scenario, condition) pair.
# Five repetitions provide a basic measure of run-to-run variability while
# keeping the total API-call budget manageable.
SCENARIOS_PER_CONDITION = 5

# Minimum sleep time (seconds) injected between consecutive API calls to
# avoid exceeding the rate limit.
API_CALL_DELAY = 1.0

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_llm(bind_tools=None):
    """Construct and return a LangChain chat model instance.

    Supports multiple backends selected via the ``LLM_MODEL`` constant:
    OpenAI (GPT-4o-mini, GPT-5.4-mini) and Ollama (Qwen2.5-7B local,
    Gemma4-31B remote).  Lazy imports avoid loading unnecessary SDKs.

    Parameters
    ----------
    bind_tools : list[Tool] | None
        If provided, the returned LLM is pre-configured with the given tools
        via LangChain's ``bind_tools`` mechanism.  This causes the model to
        emit structured ``tool_call`` objects in its responses, which the
        agent's reasoning loop then dispatches to the correct Python function.
        When ``None``, a plain chat model is returned (used by AgentManager
        for intent classification, which does not require tool access).

    Returns
    -------
    ChatOpenAI | ChatOllama
        A configured LLM instance, optionally with tools bound.
    """
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
        # bind_tools tells the LLM which functions it may call and how to
        # serialise their parameters.  LangChain handles the function-calling
        # schema generation automatically from the @tool decorators.
        return llm.bind_tools(bind_tools)
    return llm
