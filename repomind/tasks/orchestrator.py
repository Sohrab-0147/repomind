"""Plan-and-execute task orchestrator using LangGraph StateGraph."""
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from repomind.agent.tools import search_codebase
from repomind.config import load_config
from repomind.observability.logger import get_logger
from repomind.tools.filesystem_tools import list_directory, read_file
from repomind.tools.terminal_tools import run_command

logger = get_logger(__name__)
config = load_config()

CODEBASE_CONTEXT = """RepoMind is a RAG-powered code assistant. Key modules:
- repomind/config.py — loads config.yaml and resolves env vars
- repomind/context/indexers/code_parser.py — Tree-sitter AST chunking
- repomind/context/indexers/semantic_qdrant.py — Qdrant hybrid indexing
- repomind/context/retrievers/semantic_qdrant.py — hybrid retrieval + FlashRank rerank
- repomind/tools/filesystem_tools.py — sandbox enforced by _safe_resolve()
- repomind/tools/terminal_tools.py — BLOCKED_COMMANDS set, 30s timeout
- repomind/agent/factory.py — LangChain agent with tools + guardrails
- repomind/mcp/client.py — MCP GitHub integration
- repomind/memory/short_term.py — SQLite checkpointer
Note: there is no 'Sandbox' class. Sandboxing = _safe_resolve + BLOCKED_COMMANDS.
"""


class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered list of 3 concrete steps to achieve the goal")


class TaskState(TypedDict):
    goal: str
    steps: list[str]
    current: int
    results: Annotated[list[str], lambda a, b: a + b]
    final: str


def _model():
    provider = config["llm"]["provider"]
    model_name = config["llm"]["model"]
    return init_chat_model(f"{provider}:{model_name}", temperature=0)


def plan_node(state: TaskState) -> dict:
    llm = _model().with_structured_output(Plan)
    prompt = (
        f"You are planning a task for a code assistant. "
        f"Break the goal into EXACTLY 3 concrete steps. "
        f"Each step must be answerable with a single search or file read.\n\n"
        f"CODEBASE CONTEXT:\n{CODEBASE_CONTEXT}\n\n"
        f"Goal: {state['goal']}"
    )
    plan = llm.invoke(prompt)
    logger.info(f"Planned {len(plan.steps)} steps")
    return {"steps": plan.steps[:3], "current": 0, "results": []}


def execute_node(state: TaskState) -> dict:
    step = state["steps"][state["current"]]
    logger.info(f"Executing step {state['current']+1}/{len(state['steps'])}: {step}")

    agent = create_agent(
        model=_model(),
        tools=[search_codebase, read_file],
        system_prompt=(
            "You execute ONE step of a larger plan. "
            "Use search_codebase ONCE. If results are relevant, answer immediately. "
            "Do NOT search more than twice. Report only what the tools returned."
        ),
    )

    prior = ""
    if state["results"]:
        prior = "\n\nPrior step findings:\n" + state["results"][-1]

    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Step: {step}{prior}"}]
    })
    output = result["messages"][-1].content

    return {
        "current": state["current"] + 1,
        "results": [f"Step {state['current']+1}: {step}\nFindings: {output}"],
    }


def synthesize_node(state: TaskState) -> dict:
    llm = _model()
    prompt = (
        f"Goal: {state['goal']}\n\n"
        f"Findings from steps:\n" + "\n\n".join(state["results"]) + "\n\n"
        "Write a concise final answer grounded in the findings. Cite files where relevant."
    )
    result = llm.invoke(prompt)
    return {"final": result.content}


def should_continue(state: TaskState) -> str:
    if state["current"] >= len(state["steps"]):
        return "synthesize"
    return "execute"


def build_graph():
    g = StateGraph(TaskState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", should_continue, {
        "execute": "execute",
        "synthesize": "synthesize",
    })
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=MemorySaver())


def run_plan(goal: str) -> str:
    graph = build_graph()
    result = graph.invoke(
        {"goal": goal, "steps": [], "current": 0, "results": [], "final": ""},
        {"configurable": {"thread_id": "plan-" + goal[:20]}},
    )
    return result["final"]
