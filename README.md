# Advanced RAG Pipeline for AI-Driven Test Automation

A **production-grade Retrieval-Augmented Generation (RAG)** pipeline applied to academic literature on AI testing and test automation.

This project demonstrates how to build a complete, end-to-end RAG system integrating multiple advanced techniques into a single coherent pipeline.

---

## What is RAG? (Retrieval-Augmented Generation)

RAG combines two pillars:

- **Retrieval** — Fetch relevant documents from a knowledge base (here: your PDF papers).
- **Generation** — An LLM answers using those documents as context.

The model is therefore **grounded** in the indexed papers instead of relying only on its internal knowledge: it answers from your corpus, reducing hallucination and keeping answers tied to the literature.

---

## Pipeline Architecture

```
arXiv PDFs
    │
    ▼
[Semantic Chunking]          ← meaning-aware text splitting
    │
    ▼
[Chroma Vector Store]        ← persistent dense index
    │
    ▼
[Hybrid Retrieval]           ← BM25 (sparse) + Chroma (dense) via EnsembleRetriever
    │
    ▼
[Cross-Encoder Re-ranking]   ← BAAI/bge-reranker-base, keeps top 3
    │
    ▼
[Agentic RAG - LangGraph]    ← grade → rewrite → generate loop
    │
    ▼
[Gradio Chat UI]             ← streaming interface
```

---

## Techniques Implemented

| # | Technique | Idea | Why it helps |
|---|-----------|------|--------------|
| 1 | **Semantic Chunking** | Split text at **meaning boundaries** (embedding similarity between consecutive sentences), not at fixed character counts. | Papers have clear conceptual sections; cutting mid-concept hurts retrieval quality. |
| 2 | **Hybrid Retrieval** | Combine **BM25** (lexical/keyword search) and **dense** retrieval (embeddings in Chroma). | Technical terms (e.g. "Playwright", "mutation testing") benefit from BM25; semantic questions ("challenges in automated testing") from embeddings. Together (e.g. via RRF) improve recall. |
| 3 | **Cross-Encoder Re-ranking** | A second model scores (query, document) pairs for relevance. | Bi-encoders (Chroma) are fast but approximate; the cross-encoder is more accurate but slower. Using it only on a small candidate set (e.g. top 3) balances quality and cost. |
| 4 | **Parent-Document Retriever** | Index **small chunks** (children) for retrieval, but return **larger chunks** (parents) that contain them to the LLM. | Small chunks → better embedding precision; large chunks → more context for the answer. |
| 5 | **Agentic RAG (LangGraph)** | Graph: Agent → (optional) Retrieve → Grade → if not relevant Rewrite and loop back to Agent; if relevant → Generate. | The agent decides whether to search or answer; a "grader" checks if documents are useful; if not, the query is rewritten (up to N times) to improve retrieval. |
| 6 | **Gradio Streaming UI** | Interactive chat with real-time token streaming. | Better UX for long answers. |
| 7 | **Multimodal QA** | PDF pages as images for vision-LLM reasoning. | Optional path for figures and layout. |

---

## Overall Pipeline Flow

1. **Ingest** — Download PDFs from arXiv → load as `Document`s (one per page).
2. **Chunking** — Merge pages and apply **semantic split** (or recursive split) → chunks.
3. **Indexing** — Embed chunks → Chroma; optionally build parent/child structure.
4. **Retrieval** — For each question: hybrid (BM25 + Chroma) → merge (e.g. RRF) → rerank (cross-encoder) → top K.
5. **Agent** — LLM can call the retriever; retrieved docs are **graded**; if not relevant, **rewrite** query and retry (up to max); otherwise **generate** answer from context.
6. **UI** — Gradio uses the same retriever in a classic chain (retrieve + stuff documents + LLM) with streaming.

---

## Codebase Overview (Moduli)

| Module | Role |
|--------|------|
| **config.py** | Paths (`DATA_DIR`, `CHROMA_DIR`), API keys (OpenRouter, OpenAI), retrieval params (BM25_K, DENSE_K, RERANKER_TOP_N, ensemble weights), chunking params (semantic / parent / child), arXiv queries, `MAX_REWRITES`. Factory functions: `get_llm`, `get_embedding_model`. |
| **ingest.py** | `download_arxiv_papers`: search arXiv for configured queries, save PDFs under `data/`, skip already-downloaded files. `load_documents`: load each PDF with PyPDFLoader → list of `Document`s (one per page). `ingest()` = download + load. |
| **chunking.py** | `recursive_split`: `RecursiveCharacterTextSplitter` with separators `\n\n`, `\n`, `. `, space; fixed-size chunks with overlap; returns `List[Document]`. `semantic_split`: merge all pages into one text, run `SemanticChunker` (embeddings + similarity threshold to find breaks); returns `List[str]`. |
| **retrieval.py** | Builds Chroma vectorstore, BM25 index, **EnsembleRetriever** (BM25 + dense), optional cross-encoder reranker and parent-document wrapper. Exposes the retriever used by both the agent and the UI. |
| **agent.py** | **State**: `messages` + `number_of_rewrites`. **Nodes**: *agent* (LLM with "retrieve" tool; decides tool vs direct answer; after `MAX_REWRITES` answers without tool), *retrieve* (ToolNode running the retriever), *grade_documents* (LLM with structured yes/no: are docs relevant?), *rewrite* (rephrase query), *generate* (final answer from context). **Graph**: START → agent → (if tool) retrieve → grade → generate (end) or rewrite → back to agent. `build_graph(reranking_retriever)` builds the graph; `run(graph, question)` streams and returns the final answer. |
| **ui.py** | Builds `create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))`; streams by iterating `rag_chain.stream({"input": message})` and yielding accumulated `answer`. `launch(retriever, share)` starts the Gradio ChatInterface with examples and theme. |

---

## Domain

Knowledge base: automatically downloaded arXiv papers on:

- LLM-based test case generation
- AI-driven GUI testing
- Autonomous test agents
- Neural network approaches to software QA

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/rag-ai-testing.git
cd rag-ai-testing
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp env.example .env
# Edit .env and add your API keys
```

You need:

- **OpenRouter API key** → [openrouter.ai](https://openrouter.ai/) (for LLM)
- **OpenAI API key** → [platform.openai.com](https://platform.openai.com/) (for embeddings)

### 3. Run

```bash
# Launch the Gradio chat UI
python main.py

# Ask a single question directly from the terminal
python main.py --query "How can LLMs generate test cases automatically?"

# Skip re-downloading papers if already cached in ./data/
python main.py --no-download
```

On first run the pipeline will:

1. Download ~12 arXiv papers into `./data/`
2. Download the cross-encoder model (~500MB) from HuggingFace
3. Build the Chroma vector store in `./chroma/`

Subsequent runs reuse all cached data automatically.

---

## Project Structure

```
rag-ai-testing/
├── main.py              # CLI entry point
├── rag/
│   ├── __init__.py
│   ├── config.py        # LLM/embedding factory & global parameters
│   ├── ingest.py        # arXiv download + PDF loading
│   ├── chunking.py      # Recursive and semantic text splitting
│   ├── retrieval.py     # Vectorstore, BM25, hybrid, reranker, parent-doc
│   ├── agent.py         # LangGraph agentic RAG workflow
│   └── ui.py            # Gradio streaming chat interface
├── requirements.txt
├── env.example
├── .gitignore
├── data/                # Downloaded PDFs (git-ignored)
└── chroma/              # Vector store (git-ignored)
```

---

## Usage

```bash
# Launch the Gradio chat UI
python main.py

# Ask a single question from the terminal
python main.py --query "How can LLMs generate test cases automatically?"

# Skip re-downloading papers (use cached PDFs)
python main.py --no-download

# Create a public shareable Gradio link
python main.py --share
```

---

## Key Design Decisions

**Why semantic chunking over recursive splitting?**  
Technical papers have clear conceptual sections. Semantic chunking respects these boundaries, reducing the chance of splitting mid-concept.

**Why hybrid retrieval?**  
Technical terms like `Playwright`, `mutation testing`, or `test oracle` benefit from exact BM25 matching. Semantic queries like "challenges in automated testing" benefit from dense retrieval. Combining both via RRF outperforms either alone.

**Why cross-encoder re-ranking?**  
Bi-encoder embeddings (used in Chroma) trade accuracy for speed. The cross-encoder attends to both query and document jointly, significantly improving precision at the cost of higher latency — acceptable for RAG since we only re-rank a small candidate set.

**Why LangGraph for the agent?**  
LangGraph provides explicit state management and conditional edges, making the retrieve→grade→rewrite loop transparent and debuggable compared to opaque agent frameworks.

---

## Related Projects

- [ai-test-automation](https://github.com/Sere-eng/ai-test-automation) — AI Agent + Playwright for enterprise web testing
- [zabbix](https://github.com/Sere-eng/zabbix) — MCP + A2A protocol agent for Zabbix monitoring
