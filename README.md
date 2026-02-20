\# Advanced RAG Pipeline for AI-Driven Test Automation



A \*\*production-grade Retrieval-Augmented Generation (RAG)\*\* pipeline applied to academic literature on AI testing and test automation.



This project demonstrates how to build a complete, end-to-end RAG system integrating multiple advanced techniques into a single coherent pipeline.



---



\## Pipeline Architecture



```

arXiv PDFs

&nbsp;   │

&nbsp;   ▼

\[Semantic Chunking]          ← meaning-aware text splitting

&nbsp;   │

&nbsp;   ▼

\[Chroma Vector Store]        ← persistent dense index

&nbsp;   │

&nbsp;   ▼

\[Hybrid Retrieval]           ← BM25 (sparse) + Chroma (dense) via EnsembleRetriever

&nbsp;   │

&nbsp;   ▼

\[Cross-Encoder Re-ranking]   ← BAAI/bge-reranker-base, keeps top 3

&nbsp;   │

&nbsp;   ▼

\[Agentic RAG - LangGraph]    ← grade → rewrite → generate loop

&nbsp;   │

&nbsp;   ▼

\[Gradio Chat UI]             ← streaming interface

```



---



\## Techniques Implemented



| # | Technique | Description |

|---|-----------|-------------|

| 1 | \*\*Semantic Chunking\*\* | Splits text at meaning boundaries using embedding similarity (SemanticChunker) |

| 2 | \*\*Hybrid Retrieval\*\* | Combines BM25 keyword search with dense vector search via EnsembleRetriever |

| 3 | \*\*Cross-Encoder Re-ranking\*\* | Re-scores (query, document) pairs jointly for higher precision |

| 4 | \*\*Parent-Document Retriever\*\* | Small chunks for embedding quality, large chunks for LLM context |

| 5 | \*\*Agentic RAG (LangGraph)\*\* | Reactive agent with document grading, query rewriting, and adaptive generation |

| 6 | \*\*Gradio Streaming UI\*\* | Interactive chat interface with real-time token streaming |

| 7 | \*\*Multimodal QA\*\* | PDF pages rendered as images for vision-LLM reasoning |



---



\## Domain



Knowledge base: automatically downloaded arXiv papers on:

\- LLM-based test case generation

\- AI-driven GUI testing

\- Autonomous test agents

\- Neural network approaches to software QA



---



\## Setup



\### 1. Clone and install dependencies



```bash

git clone https://github.com/YOUR\_USERNAME/rag-ai-testing.git

cd rag-ai-testing

pip install -r requirements.txt

```



\### 2. Configure environment variables



```bash

cp .env.example .env

\# Edit .env and add your API keys

```



You need:

\- \*\*OpenRouter API key\*\* → \[openrouter.ai](https://openrouter.ai/) (for LLM)

\- \*\*OpenAI API key\*\* → \[platform.openai.com](https://platform.openai.com/) (for embeddings)



\### 3. Run



```bash

\# Launch the Gradio chat UI

python main.py



\# Ask a single question directly from the terminal

python main.py --query "How can LLMs generate test cases automatically?"



\# Skip re-downloading papers if already cached in ./data/

python main.py --no-download

```



On first run the pipeline will:

1\. Download ~12 arXiv papers into `./data/`

2\. Download the cross-encoder model (~500MB) from HuggingFace

3\. Build the Chroma vector store in `./chroma/`



Subsequent runs reuse all cached data automatically.



---



\## Project Structure



```

rag-ai-testing/

├── main.py              # CLI entry point

├── rag/

│   ├── \_\_init\_\_.py

│   ├── config.py        # LLM/embedding factory functions \& global parameters

│   ├── ingest.py        # arXiv download + PDF loading

│   ├── chunking.py      # Recursive and semantic text splitting

│   ├── retrieval.py     # Vectorstore, BM25, hybrid, reranker, parent-doc

│   ├── agent.py         # LangGraph agentic RAG workflow

│   └── ui.py            # Gradio streaming chat interface

├── requirements.txt

├── .env.example

├── .gitignore

├── data/                # Downloaded PDFs (git-ignored)

└── chroma/              # Vector store (git-ignored)

```



\## Usage



```bash

\# Launch the Gradio chat UI

python main.py



\# Ask a single question from the terminal

python main.py --query "How can LLMs generate test cases automatically?"



\# Skip re-downloading papers (use cached PDFs)

python main.py --no-download



\# Create a public shareable Gradio link

python main.py --share

```



---



\## Key Design Decisions



\*\*Why semantic chunking over recursive splitting?\*\*

Technical papers have clear conceptual sections. Semantic chunking respects these boundaries, reducing the chance of splitting mid-concept.



\*\*Why hybrid retrieval?\*\*

Technical terms like `Playwright`, `mutation testing`, or `test oracle` benefit from exact BM25 matching. Semantic queries like "challenges in automated testing" benefit from dense retrieval. Combining both via RRF outperforms either alone.



\*\*Why cross-encoder re-ranking?\*\*

Bi-encoder embeddings (used in Chroma) trade accuracy for speed. The cross-encoder attends to both query and document jointly, significantly improving precision at the cost of higher latency — acceptable for RAG since we only re-rank a small candidate set.



\*\*Why LangGraph for the agent?\*\*

LangGraph provides explicit state management and conditional edges, making the retrieve→grade→rewrite loop transparent and debuggable compared to opaque agent frameworks.



---



\## Related Projects



\- \[ai-test-automation](https://github.com/Sere-eng/ai-test-automation) — AI Agent + Playwright for enterprise web testing

\- \[zabbix](https://github.com/Sere-eng/zabbix) — MCP + A2A protocol agent for Zabbix monitoring



---

