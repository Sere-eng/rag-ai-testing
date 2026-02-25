"""
main.py
-------
CLI entry point for the RAG pipeline.

Usage:
    # Build the pipeline and launch the Gradio UI
    python main.py

    # Ask a single question from the command line
    python main.py --query "How can LLMs generate test cases automatically?"

    # Skip download if papers are already cached
    python main.py --no-download
"""

from __future__ import annotations

import argparse

from rag.agent import build_graph, run
from rag.chunking import semantic_split
from rag.ingest import ingest
from rag.retrieval import (
    build_hybrid_retriever,
    build_reranking_retriever,
    build_vectorstore,
)
from rag import ui


def build_pipeline(skip_download: bool = False):
    """
    Run the full pipeline setup:
      1. Ingest papers (download + load)
      2. Semantic chunking
      3. Build Chroma vectorstore
      4. Build hybrid retriever (BM25 + dense)
      5. Add cross-encoder reranker

    Returns:
        (reranking_retriever, graph)
    """
    print("\n" + "="*60)
    print(" RAG Pipeline — AI-Driven Test Automation")
    print("="*60 + "\n")

    # 1. Ingest
    documents = ingest()

    # 2. Chunk
    print("\n[2/4] Chunking documents...")
    chunks = semantic_split(documents)

    # 3. Vectorstore
    print("\n[3/4] Building vector store...")
    vectorstore = build_vectorstore(chunks)

    # 4. Retrieval stack
    print("\n[4/4] Building retrieval stack...")
    hybrid    = build_hybrid_retriever(chunks, vectorstore)
    reranker  = build_reranking_retriever(hybrid)

    # 5. Agent
    graph = build_graph(reranker)

    return reranker, graph


def main():
    parser = argparse.ArgumentParser(
        description="Advanced RAG pipeline for AI testing papers"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Ask a single question and print the answer (skips Gradio UI)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip arXiv download (use cached PDFs in ./data/)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    args = parser.parse_args()

    reranker, graph = build_pipeline(skip_download=args.no_download)

    if args.query:
        # Single-question mode
        print(f"\n{'='*60}")
        print(f"QUESTION: {args.query}")
        print("="*60)
        answer = run(graph, args.query)
        print(f"\nANSWER:\n{answer}\n")
    else:
        # Interactive Gradio UI (agentic)
        print("\n✓ Launching Gradio interface...")
        ui.launch(graph, share=args.share)


if __name__ == "__main__":
    main()