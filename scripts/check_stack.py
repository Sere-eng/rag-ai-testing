#!/usr/bin/env python3
"""
Verifica che lo stack (import, config, retrieval) sia ok dopo aggiornamenti.
Esegui dopo: pip install -r requirements.txt --upgrade

Uso:
  python scripts/check_stack.py
  python scripts/check_stack.py --quick   # solo import, niente retrieval
"""
from __future__ import annotations

import argparse
import sys


def check_imports() -> list[str]:
    """Verifica che tutti i moduli critici si importino. Ritorna lista errori (vuota se ok)."""
    errors = []

    # 1. Config e env
    try:
        from rag import config
        assert hasattr(config, "get_embedding_model") and hasattr(config, "get_llm")
    except Exception as e:
        errors.append(f"rag.config: {e}")
        return errors  # il resto dipende da config

    # 2. Retrieval (include ContextualCompressionRetriever e CrossEncoderReranker)
    try:
        from rag.retrieval import (
            build_hybrid_retriever,
            build_reranking_retriever,
            build_vectorstore,
            ContextualCompressionRetriever,
            CrossEncoderReranker,
        )
    except Exception as e:
        errors.append(f"rag.retrieval: {e}")
        return errors

    # 3. Da dove viene ContextualCompressionRetriever
    try:
        from langchain.retrievers import ContextualCompressionRetriever as Official
        from rag.retrieval import ContextualCompressionRetriever as Used
        source = "langchain.retrievers (ufficiale)" if Used is Official else "fallback locale"
    except ImportError:
        source = "fallback locale (langchain.retrievers non disponibile)"
    print(f"  ContextualCompressionRetriever: {source}")

    # 4. Agent
    try:
        from rag.agent import build_graph, run
    except Exception as e:
        errors.append(f"rag.agent: {e}")

    # 5. Chunking e ingest
    try:
        from rag.chunking import semantic_split
        from rag.ingest import ingest
    except Exception as e:
        errors.append(f"rag.chunking/ingest: {e}")

    return errors


def check_retrieval_mini() -> list[str]:
    """Verifica minima pipeline retrieval (chunks finti, Chroma in temp). Ritorna lista errori."""
    errors = []

    try:
        import tempfile
        from rag.retrieval import build_hybrid_retriever, build_vectorstore, build_reranking_retriever
    except Exception as e:
        errors.append(f"Import retrieval: {e}")
        return errors

    # Chunks finti; Chroma in directory temporanea per non toccare chroma reale
    fake_chunks = [
        "LLMs can generate test cases by using natural language prompts.",
        "Automated testing with AI involves mutation testing and oracles.",
    ]

    try:
        with tempfile.TemporaryDirectory(prefix="rag_check_") as tmp:
            vs = build_vectorstore(fake_chunks, persist_directory=tmp)
            hybrid = build_hybrid_retriever(fake_chunks, vs)
            # Reranker carica il modello (~500MB) solo qui
            reranking = build_reranking_retriever(hybrid)
            docs = reranking.invoke("How do LLMs generate tests?")
        if not isinstance(docs, list):
            errors.append("reranking.invoke() non ha restituito una lista")
        else:
            print(f"  Retrieval mini-test: ok (recuperati {len(docs)} doc)")
    except Exception as e:
        errors.append(f"Retrieval mini-test: {e}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlli stack RAG dopo aggiornamenti")
    parser.add_argument("--quick", action="store_true", help="Solo import, senza test retrieval")
    args = parser.parse_args()

    print("Controlli stack RAG\n")

    all_errors = []

    print("[1/2] Import moduli...")
    err = check_imports()
    if err:
        all_errors.extend(err)
    else:
        print("  Import: ok\n")

    if not args.quick:
        print("[2/2] Test retrieval (chunks finti, cross-encoder scaricato se necessario)...")
        err = check_retrieval_mini()
        if err:
            all_errors.extend(err)
    else:
        print("[2/2] Saltato (--quick)\n")

    if all_errors:
        print("Errori:")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("Tutti i controlli passati.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
