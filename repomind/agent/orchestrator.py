from collections.abc import Iterator

from repomind.agent.factory import build_agent
from repomind.observability.logger import get_logger

logger = get_logger(__name__)


def handle_query_stream(question: str, thread_id: str) -> Iterator[dict]:
    """Yield events: {'type': 'tool', 'name': ...} or {'type': 'text', 'content': ...}"""
    logger.info(f"Handling query for session {thread_id}: {question}")

    agent = build_agent()
    agent_config = {"configurable": {"thread_id": thread_id}}

    for chunk, _meta in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        agent_config,
        stream_mode="messages",
    ):
        name = type(chunk).__name__

        if name == "AIMessageChunk":
            tool_chunks = getattr(chunk, "tool_call_chunks", None)
            if tool_chunks:
                for tc in tool_chunks:
                    tname = tc.get("name")
                    if tname:
                        yield {"type": "tool", "name": tname}
                continue

            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if content:
                yield {"type": "text", "content": content}

        elif name == "ToolMessage":
            yield {"type": "tool_done", "name": getattr(chunk, "name", "")}


def handle_query(question: str, thread_id: str) -> str:
    parts = []
    for ev in handle_query_stream(question, thread_id):
        if ev.get("type") == "text":
            parts.append(ev["content"])
    return "".join(parts) if parts else "No response generated."
