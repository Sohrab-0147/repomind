import asyncio
import threading

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from repomind.agent.tools import search_codebase
from repomind.llm.factory import get_llm
from repomind.mcp.client import get_mcp_tools
from repomind.memory.short_term import get_checkpointer
from repomind.observability.logger import get_logger
from repomind.tools.filesystem_tools import list_directory, read_file
from repomind.tools.terminal_tools import run_command

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.

Tools available:
- search_codebase: semantic search across the codebase (use ONCE per question)
- read_file, list_directory, run_command: sandboxed filesystem and shell
- GitHub tools (list_issues, create_issue, get_pull_request, list_commits, search_repositories, get_file_contents)

RULES:
1. Search codebase ONCE with a clear query.
2. If results are relevant, ANSWER IMMEDIATELY.
3. Do NOT search multiple times with similar queries.
4. Cite file names and line numbers.
5. Use GitHub tools when the question is about issues, PRs, or commits.
6. If you cannot find the answer, say so."""

KEEP_MCP_TOOLS = {
    "list_issues",
    "create_issue",
    "get_pull_request",
    "list_commits",
    "search_repositories",
    "get_file_contents",
}

# Dedicated event loop on a background thread for sync-wrapping async MCP tools
_loop = asyncio.new_event_loop()
_thr = threading.Thread(target=_loop.run_forever, name="mcp-async-runner", daemon=True)
_thr.start()


def _run_async(coro):
    """Run an async coroutine from sync code using the background event loop."""
    if not _thr.is_alive():
        _thr.start()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


def _make_sync(tool):
    """Wrap an async-only StructuredTool with a sync func."""
    if not isinstance(tool, StructuredTool):
        return tool
    if tool.func is not None or tool.coroutine is None:
        return tool

    def _sync_func(*args, **kwargs):
        return _run_async(tool.coroutine(*args, **kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_sync_func,
        coroutine=tool.coroutine,
    )


def build_agent():
    llm = get_llm()
    checkpointer = get_checkpointer()

    try:
        all_mcp = asyncio.run(get_mcp_tools())
        filtered = [t for t in all_mcp if t.name in KEEP_MCP_TOOLS]
        mcp_tools = [_make_sync(t) for t in filtered]
        logger.info(f"Filtered MCP tools: {len(mcp_tools)} of {len(all_mcp)}")
    except Exception as e:
        logger.warning(f"MCP tools unavailable: {e}")
        mcp_tools = []

    tools = [search_codebase, read_file, list_directory, run_command] + mcp_tools
    logger.info(f"Creating agent with {len(tools)} tools")

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
