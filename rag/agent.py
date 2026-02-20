"""
agent.py
--------
Agentic RAG workflow built with LangGraph.

Graph structure:
    START → agent → retrieve → grade_documents → generate → END
                        ↑           ↓ (not relevant)
                     rewrite ←──────┘

- agent          : decides whether to call the retriever tool or answer directly
- retrieve       : calls the retriever tool (hybrid + reranking)
- grade_documents: binary relevance check on retrieved docs
- rewrite        : rephrases the query to improve retrieval (max MAX_REWRITES times)
- generate       : produces the final answer from relevant context
"""

from __future__ import annotations

from typing import Annotated, Literal, Sequence

from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rag.config import MAX_REWRITES, get_llm


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_rewrites: int


# ── Node & edge definitions ────────────────────────────────────────────────────

def make_agent_node(tools: list):
    """Returns the agent node function bound to the given tools."""

    def agent(state: AgentState) -> dict:
        """Decides whether to call a tool or answer directly."""
        print("--- AGENT ---")
        llm = get_llm()

        if state["number_of_rewrites"] < MAX_REWRITES:
            llm = llm.bind_tools(tools)
        else:
            print("  → Max rewrites reached, answering from own knowledge")

        response = llm.invoke(state["messages"])
        return {"messages": [response], "number_of_rewrites": state["number_of_rewrites"]}

    return agent


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """
    Grades the relevance of retrieved documents.
    Returns 'generate' if relevant, 'rewrite' otherwise.
    """
    print("--- GRADING DOCUMENTS ---")

    class GradeScore(BaseModel):
        """Binary relevance score for a retrieved document."""
        binary_score: str = Field(description="'yes' if document is relevant, 'no' otherwise")

    grader_llm = get_llm(temperature=0).with_structured_output(GradeScore)

    grade_prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "Document:\n{context}\n\n"
            "Question: {question}\n"
            "Give a binary score 'yes' or 'no'."
        ),
        input_variables=["context", "question"],
    )

    question = state["messages"][0].content
    docs = state["messages"][-1].content

    result = (grade_prompt | grader_llm).invoke({"question": question, "context": docs})

    if result.binary_score == "yes":
        print("  → Documents RELEVANT — generating answer")
        return "generate"
    else:
        print("  → Documents NOT RELEVANT — rewriting query")
        return "rewrite"


def rewrite(state: AgentState) -> dict:
    """Rewrites the query to improve retrieval on the next attempt."""
    print("--- REWRITING QUERY ---")
    state["number_of_rewrites"] += 1

    question = state["messages"][0].content
    rewrite_llm = get_llm(temperature=0.7)

    msg = HumanMessage(
        content=(
            f"Reformulate this question to improve document retrieval.\n"
            f"Be more specific and use different wording.\n\n"
            f"Original question: {question}\n\n"
            f"Improved question:"
        )
    )
    response = rewrite_llm.invoke([msg])
    print(f"  → Rewritten ({state['number_of_rewrites']}): {response.content[:100]}")
    return {"messages": [response], "number_of_rewrites": state["number_of_rewrites"]}


def generate(state: AgentState) -> dict:
    """Generates the final answer using retrieved context."""
    print("--- GENERATING ---")

    question = state["messages"][0].content
    context  = state["messages"][-1].content

    gen_prompt = PromptTemplate(
        template=(
            "You are an expert assistant on AI-driven software testing.\n"
            "Answer the question using ONLY the provided context from academic papers.\n"
            "If the context doesn't contain the answer, say so clearly.\n"
            "Be concise and precise (3-5 sentences max).\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    gen_llm = get_llm(temperature=0.3)
    response = (gen_prompt | gen_llm).invoke({"context": context, "question": question})
    return {"messages": [response], "number_of_rewrites": state["number_of_rewrites"]}


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(reranking_retriever):
    """
    Compile the LangGraph workflow.

    Args:
        reranking_retriever: The full retrieval pipeline (hybrid + reranker).

    Returns:
        Compiled LangGraph graph.
    """
    retriever_tool = create_retriever_tool(
        reranking_retriever,
        name="retrieve_ai_testing_papers",
        description=(
            "Search and retrieve information from academic papers on "
            "AI-driven test automation, LLM-based testing, and autonomous test agents. "
            "Use this tool for any question about software testing with AI."
        ),
    )
    tools = [retriever_tool]

    workflow = StateGraph(AgentState)

    workflow.add_node("agent",    make_agent_node(tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite",  rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent",    tools_condition, {"tools": "retrieve", END: END})
    workflow.add_conditional_edges("retrieve", grade_documents, {"generate": "generate", "rewrite": "rewrite"})
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite",  "agent")

    graph = workflow.compile()
    print("✓ LangGraph workflow compiled")
    return graph


# ── Public API ─────────────────────────────────────────────────────────────────

def run(graph, question: str) -> str:
    """
    Run the agentic RAG pipeline for a single question.

    Args:
        graph: Compiled LangGraph graph.
        question: Natural language question.

    Returns:
        Final answer string.
    """
    inputs = {
        "messages": [("human", question)],
        "number_of_rewrites": 0,
    }

    final_answer = ""
    for output in graph.stream(inputs):
        for _, value in output.items():
            if "messages" in value and value["messages"]:
                last = value["messages"][-1]
                if hasattr(last, "content") and last.content:
                    final_answer = last.content

    return final_answer