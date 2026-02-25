"""
ui.py
-----
Gradio chat interface backed by the LangGraph agent.
"""

from __future__ import annotations

from typing import Dict, List

import gradio as gr

from rag.agent import run

EXAMPLES: List[List[str]] = [
    ["How can LLMs generate test cases automatically?"],
    ["What are the advantages of AI-based GUI testing over traditional approaches?"],
    ["How do autonomous agents handle the test oracle problem?"],
    ["What metrics are used to evaluate AI test generation quality?"],
]


def make_agent_fn(graph):
    """
    Returns a Gradio-compatible function that calls the LangGraph agent.
    One call per user message; returns the final answer string.
    """

    def respond(message: str, history: List[Dict]) -> str:
        return run(graph, message)

    return respond


def launch(graph, share: bool = False) -> None:
    """
    Build and launch the Gradio chat interface.

    Args:
        graph     : Compiled LangGraph agent graph to run per message.
        share     : If True, creates a public Gradio link.
    """
    gr.close_all()

    demo = gr.ChatInterface(
        fn=make_agent_fn(graph),
        title="AI Testing Knowledge Assistant",
        description=(
            "Ask questions about AI-driven test automation, LLM-based testing, "
            "and autonomous test agents — powered by academic papers."
        ),
        # Gradio >=5 non accetta più i parametri `type` e `theme`
        # nel costruttore di ChatInterface.
        examples=EXAMPLES,
    )

    demo.launch(share=share)