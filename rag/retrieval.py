"""
chunking.py
-----------
Text splitting strategies:
  - RecursiveCharacterTextSplitter  (fast, size-based baseline)
  - SemanticChunker                 (meaning-aware, used in main pipeline)
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    SEMANTIC_BREAKPOINT,
    SEMANTIC_MIN_CHUNK,
    get_embedding_model,
)


def recursive_split(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Baseline splitting: fixed-size chunks with overlap.
    Fast and deterministic — useful for comparison.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Recursive split → {len(chunks)} chunks")
    return chunks


def semantic_split(documents: List[Document]) -> List[str]:
    """
    Semantic splitting: splits at meaning boundaries detected via
    embedding cosine similarity drops between consecutive sentences.

    Returns plain strings (SemanticChunker works on raw text).
    Why plain text? SemanticChunker fuses all pages into one stream,
    then cuts at semantic breaks — metadata is intentionally discarded.
    """
    embedding_model = get_embedding_model()
    splitter = SemanticChunker(
        embeddings=embedding_model,
        min_chunk_size=SEMANTIC_MIN_CHUNK,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT,
    )

    # Join all pages into a single text stream for cross-page coherence
    full_text = "\n\n".join(
        doc.page_content for doc in documents if doc.page_content.strip()
    )
    chunks = splitter.split_text(full_text)
    print(f"Semantic split  → {len(chunks)} chunks")
    return chunks