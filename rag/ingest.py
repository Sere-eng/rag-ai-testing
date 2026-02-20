"""
ingest.py
---------
Download arXiv papers and load them as LangChain Documents.
Papers are cached locally so the download only happens once.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import arxiv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from tqdm import tqdm

from rag.config import ARXIV_QUERIES, DATA_DIR, MAX_RESULTS_PER_QUERY


def download_arxiv_papers(
    queries: List[str] = ARXIV_QUERIES,
    max_results: int = MAX_RESULTS_PER_QUERY,
    output_dir: Path = DATA_DIR,
) -> List[str]:
    """
    Search arXiv and download PDFs to output_dir.
    Skips papers that are already downloaded (filename match).

    Returns:
        List of absolute paths to all available PDFs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()
    downloaded: list[str] = []

    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for paper in tqdm(client.results(search), desc=f"arXiv: {query[:40]}"):
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_"
                for c in paper.title[:60]
            ).strip()
            pdf_path = output_dir / f"{safe_title}.pdf"

            if not pdf_path.exists():
                try:
                    paper.download_pdf(dirpath=str(output_dir), filename=pdf_path.name)
                    print(f"  ✓ Downloaded: {pdf_path.name}")
                except Exception as exc:
                    print(f"  ⚠ Could not download '{paper.title[:50]}': {exc}")
                    continue

            downloaded.append(str(pdf_path))

    # Deduplicate (same paper may appear in multiple queries)
    unique = list(dict.fromkeys(downloaded))
    print(f"\n✓ {len(unique)} PDFs available in {output_dir}")
    return unique


def load_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Load all PDFs into LangChain Document objects.

    Returns:
        Flat list of Document objects (one per page).
    """
    all_docs: list[Document] = []

    for path in tqdm(pdf_paths, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
        except Exception as exc:
            print(f"  ⚠ Could not load {path}: {exc}")

    print(f"\n✓ Loaded {len(all_docs)} pages from {len(pdf_paths)} PDFs")
    return all_docs


def ingest(
    queries: List[str] = ARXIV_QUERIES,
    max_results: int = MAX_RESULTS_PER_QUERY,
    data_dir: Path = DATA_DIR,
) -> List[Document]:
    """
    Full ingestion pipeline: download papers → load documents.
    Entry point used by the rest of the pipeline.
    """
    pdf_paths = download_arxiv_papers(queries, max_results, data_dir)
    return load_documents(pdf_paths)