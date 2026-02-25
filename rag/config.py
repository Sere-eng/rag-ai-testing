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
from langchain_huggingface import HuggingFaceEmbeddings

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

# ── Embeddings backend ─────────────────────────────────────────────────────────
# EMBEDDING_PROVIDER controls which backend is used for embeddings:
#   - "openai"      → OpenAIEmbeddings (text-embedding-3-small by default)
#   - "huggingface" → HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5 by default)
EMBEDDING_PROVIDER             = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_HF_EMBEDDING_MODEL     = "BAAI/bge-small-en-v1.5"

# ── Retrieval parameters ────────────────────────────────────────────────────────
BM25_K          = 4     # most relevant documents returned by BM25 -> sparse keyword search
DENSE_K         = 4     # most relevant documents returned by Chroma -> dense embedding search
# After the hybrid retrieval you have up to 8 documents in total (4 from BM25 + 4 from Chroma, with possible duplicates removed).
RERANKER_TOP_N  = 3     # documents kept after cross-encoder re-ranking (The cross-encoder takes those 8 documents, rearranges them by relevance, and keeps only the top 3 to pass to the LLM.)
ENSEMBLE_WEIGHTS = [0.5, 0.5]   # [BM25 weight, dense weight]

# ── Chunking parameters ─────────────────────────────────────────────────────────
SEMANTIC_MIN_CHUNK   = 500 # minimum chunk size for semantic splitting (a semantic chunk must be at least 500 characters. Without this limit, section titles would become separate chunks on their own.)
SEMANTIC_BREAKPOINT  = 0.5 # threshold for semantic splitting (a semantic chunk is created when the embedding similarity between consecutive sentences drops below 0.5.)
PARENT_CHUNK_SIZE    = 2000 # parent chunks are larger than child chunks to provide more context for the LLM (the large chunks that are returned to the LLM after the child is found.)
PARENT_CHUNK_OVERLAP = 200 # overlap between parent chunks
CHILD_CHUNK_SIZE     = 400 # the small chunks that are indexed in the vectorstore for precise search
CHILD_CHUNK_OVERLAP  = 50 # overlap between child chunks

# ── arXiv download ──────────────────────────────────────────────────────────────
ARXIV_QUERIES = [
    "LLM test automation software testing",
    "AI autonomous test generation",
    "large language model GUI testing",
    "neural network software quality assurance",
]
MAX_RESULTS_PER_QUERY = 3 # up to 3 results per query (up to 12 results total)

# ── Agent ───────────────────────────────────────────────────────────────────────
MAX_REWRITES = 3   # max query rewrites before falling back to LLM knowledge (retrieve → grade → rewrite → retrieve → grade → rewrite ...)


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


# temperature — control the creativity of the response. It goes from 0 to 1:
# 0.0 → deterministic, always the same answer (used for grader)
# 0.7 → more creative (used for rewrite)
# 0.3 → balanced (used for generation)


def get_embedding_model():
    """Returns the shared embedding model (OpenAI or Hugging Face)."""
    if EMBEDDING_PROVIDER == "huggingface":
        model_name = os.environ.get(
            "HF_EMBEDDING_MODEL_NAME",
            DEFAULT_HF_EMBEDDING_MODEL,
        )
        # normalize_embeddings=True è consigliato per la similarity search
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Default: OpenAI embeddings
    model_name = os.environ.get(
        "OPENAI_EMBEDDING_MODEL_NAME",
        DEFAULT_OPENAI_EMBEDDING_MODEL,
    )
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY,
    )