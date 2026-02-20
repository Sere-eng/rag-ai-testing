"""
config.py
---------
Central configuration: LLM, embeddings, paths, and constants.
All tunable parameters live here so the rest of the codebase stays clean.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv(), override=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma" / "ai-testing-semantic"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", OPENROUTER_API_KEY)

# ── Retrieval parameters ────────────────────────────────────────────────────────
BM25_K          = 4     # documents returned by BM25
DENSE_K         = 4     # documents returned by Chroma
RERANKER_TOP_N  = 3     # documents kept after cross-encoder re-ranking
ENSEMBLE_WEIGHTS = [0.5, 0.5]   # [BM25 weight, dense weight]

# ── Chunking parameters ─────────────────────────────────────────────────────────
SEMANTIC_MIN_CHUNK   = 500
SEMANTIC_BREAKPOINT  = 0.5
PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 50

# ── arXiv download ──────────────────────────────────────────────────────────────
ARXIV_QUERIES = [
    "LLM test automation software testing",
    "AI autonomous test generation",
    "large language model GUI testing",
    "neural network software quality assurance",
]
MAX_RESULTS_PER_QUERY = 3

# ── Agent ───────────────────────────────────────────────────────────────────────
MAX_REWRITES = 3   # max query rewrites before falling back to LLM knowledge


# ── Factory functions ───────────────────────────────────────────────────────────
def get_llm(
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    streaming: bool = False,
) -> ChatOpenAI:
    """Returns a ChatOpenAI instance pointing to OpenRouter."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        max_tokens=1024,
    )


def get_embedding_model() -> OpenAIEmbeddings:
    """Returns the shared OpenAI embedding model (text-embedding-3-small)."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )