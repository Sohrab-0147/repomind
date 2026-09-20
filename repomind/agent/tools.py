from langchain_core.tools import tool

from repomind.context.retrievers.semantic_qdrant import retrieve
from repomind.observability.logger import get_logger

logger = get_logger(__name__)


@tool
def search_codebase(query: str) -> str:
    """Search the codebase for relevant functions, classes, or logic.
    Use this whenever you need to find code related to a question."""
    logger.info(f"Tool called: search_codebase with query: {query}")

    chunks = retrieve(query, k=5)

    if not chunks:
        return "No relevant code found."

    results = []
    for chunk in chunks:
        results.append(
            f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"Type: {chunk['type']} - {chunk['name']}\n"
            f"Code:\n{chunk['content']}\n"
        )
    return "\n---\n".join(results)
