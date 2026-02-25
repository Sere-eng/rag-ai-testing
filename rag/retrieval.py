"""
retrieval.py
------------
Builds the full retrieval stack:
  1. Chroma vector store        (dense, embedding-based)
  2. BM25 retriever             (sparse, keyword-based)
  3. EnsembleRetriever          (hybrid: BM25 + Chroma via RRF)
  4. CrossEncoder re-ranker     (precision boost, keeps top-N)
  5. ParentDocumentRetriever    (small embed → large context)
"""

from __future__ import annotations

from typing import List, Optional

from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    BM25_K,
    CHROMA_DIR,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    DENSE_K,
    ENSEMBLE_WEIGHTS,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    RERANKER_TOP_N,
    get_embedding_model,
)


def build_vectorstore(chunks: List[str], persist_directory: Optional[str] = None) -> Chroma:
    """
    Build (or rebuild) a persistent Chroma vector store from semantic chunks.
    Clears any previous data before re-indexing.

    Args:
        chunks: List of text chunks to index.
        persist_directory: Optional path; if None, uses CHROMA_DIR from config.
    """
    embedding_model = get_embedding_model()
    path = persist_directory or str(CHROMA_DIR)

    vectorstore = Chroma(
        persist_directory=path,
        embedding_function=embedding_model,
    )

    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(existing_ids)
        print(f"  Cleared {len(existing_ids)} existing vectors")

    vectorstore.add_texts(chunks)
    print(f"✓ Indexed {vectorstore._collection.count()} chunks into Chroma")
    return vectorstore


def build_hybrid_retriever(
    chunks: List[str],
    vectorstore: Chroma,
) -> EnsembleRetriever:
    """
    Hybrid retriever: BM25 (sparse) + Chroma (dense), fused via RRF.

    Why hybrid?
    - BM25 excels at exact keyword matches (e.g. 'Playwright', 'mutation testing')
    - Dense retrieval excels at semantic similarity queries
    - RRF fusion outperforms either strategy alone
    """
    # BM25: lexical search — conta termini e frequenze, niente embedding
    bm25 = BM25Retriever.from_texts(chunks)
    bm25.k = BM25_K  # restituisce i 4 chunk più rilevanti per keyword

    # Dense: ricerca per similarità semantica tramite embedding
    dense = vectorstore.as_retriever(search_kwargs={"k": DENSE_K})

    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=ENSEMBLE_WEIGHTS,
    )
    print(f"✓ Hybrid retriever ready (BM25 k={BM25_K}, dense k={DENSE_K})")
    return ensemble


def build_reranking_retriever(
    ensemble: EnsembleRetriever,
    model_name: str = "BAAI/bge-reranker-base",
) -> ContextualCompressionRetriever:
    """
    Wraps the hybrid retriever with a cross-encoder re-ranker.

    The cross-encoder attends to both the query and each document jointly,
    producing a more accurate relevance score than bi-encoder embeddings.
    We keep only the top RERANKER_TOP_N documents for generation.
    """
    print(f"Loading cross-encoder model '{model_name}' (first run downloads ~500MB)...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANKER_TOP_N)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )
    print(f"✓ Re-ranking retriever ready (top_n={RERANKER_TOP_N})")
    return retriever


def build_parent_doc_retriever(documents: List[Document]) -> ParentDocumentRetriever:
    """
    Parent-Document Retriever: embeds small child chunks for accurate retrieval,
    but returns the larger parent chunk to the LLM for richer context.

    Child chunks (400 chars) → precise embedding similarity
    Parent chunks (2000 chars) → rich context for generation
    """
    embedding_model = get_embedding_model()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    child_vectorstore = Chroma(
        collection_name="parent_doc_children",
        embedding_function=embedding_model,
    )

    # InMemoryStore: dizionario in memoria che mappa ogni chunk figlio al suo padre.
    # I padri contengono il testo lungo (2000 chars), i figli hanno metadata
    # che punta all'ID del padre.
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(documents)

    n_parents  = len(list(docstore.yield_keys()))
    n_children = child_vectorstore._collection.count()
    print(f"✓ Parent-doc retriever ready ({n_parents} parents, {n_children} children)")
    return retriever