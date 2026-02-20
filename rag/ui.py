"""
ui.py
-----
Gradio chat interface with streaming responses.
Uses the full retrieval pipeline (hybrid + reranking) for each user message.
"""

from __future__ import annotations

from typing import Dict, Generator, List

import gradio as gr
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from rag.config import get_llm

SYSTEM_PROMPT = (
    "You are an expert assistant on AI-driven software testing and test automation.\n"
    "Answer questions using ONLY the provided context from academic papers.\n"
    "If the context doesn't contain the answer, say so clearly.\n"
    "Be precise and cite relevant concepts from the papers.\n\n"
    "{context}"
)

EXAMPLES = [
    ["How can LLMs generate test cases automatically?"],
    ["What are the advantages of AI-based GUI testing over traditional approaches?"],
    ["How do autonomous agents handle the test oracle problem?"],
    ["What metrics are used to evaluate AI test generation quality?"],
]


def make_streaming_fn(retriever: BaseRetriever):
    """
    Returns a Gradio-compatible streaming function that uses the given retriever.
    Each call builds a fresh RAG chain to avoid state leakage between sessions.
    """

    def respond(message: str, history: List[Dict]) -> Generator[str, None, None]:
        llm = get_llm(temperature=0.3, streaming=True)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        qa_chain  = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        full_response = ""
        for chunk in rag_chain.stream({"input": message}):
            if "answer" in chunk:
                full_response += str(chunk["answer"])
                yield full_response

    return respond


def launch(retriever: BaseRetriever, share: bool = False) -> None:
    """
    Build and launch the Gradio chat interface.

    Args:
        retriever : The retrieval pipeline to use for each query.
        share     : If True, creates a public Gradio link.
    """
    gr.close_all()

    demo = gr.ChatInterface(
        fn=make_streaming_fn(retriever),
        title="🤖 AI Testing Knowledge Assistant",
        description=(
            "Ask questions about AI-driven test automation, LLM-based testing, "
            "and autonomous test agents — powered by academic papers."
        ),
        type="messages",
        examples=EXAMPLES,
        theme=gr.themes.Soft(),
    )

    demo.launch(share=share)
    return demo