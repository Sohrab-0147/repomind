from collections.abc import Iterator

from repomind.agent.factory import build_agent
from repomind.observability.logger import get_logger

logger = get_logger(__name__)


def handle_query_stream(question: str, thread_id: str) -> Iterator[str]:
    """Stream the agent's final answer token-by-token."""
    logger.info(f"Handling query for session {thread_id}: {question}")

    agent = build_agent()
    agent_config = {"configurable": {"thread_id": thread_id}}

    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        agent_config,
        stream_mode="messages",
    ):
        # Only forward text tokens from AI message chunks
        if type(chunk).__name__ != "AIMessageChunk":
            continue
        # Skip tool-call JSON chunks
        if getattr(chunk, "tool_call_chunks", None):
            continue
        content = getattr(chunk, "content", "")
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if content:
            yield content


def handle_query(question: str, thread_id: str) -> str:
    chunks = list(handle_query_stream(question, thread_id))
    return "".join(chunks) if chunks else "No response generated."
