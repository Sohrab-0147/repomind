from langchain.agents import create_agent

from repomind.agent.tools import search_codebase
from repomind.llm.factory import get_llm
from repomind.memory.short_term import get_checkpointer
from repomind.observability.logger import get_logger
from repomind.tools.filesystem_tools import list_directory, read_file
from repomind.tools.terminal_tools import run_command

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.

Tools:
- search_codebase: semantic search across the codebase (use first)
- read_file: read a file inside the workspace
- list_directory: list files in a directory
- run_command: run shell commands (30s timeout, dangerous commands blocked)

Always search before answering. Cite file names and line numbers.
Paths are relative to the workspace root. If you cannot find the answer, say so."""


def build_agent():
    llm = get_llm()
    checkpointer = get_checkpointer()
    logger.info("Creating agent with 4 core tools + memory")

    return create_agent(
        model=llm,
        tools=[search_codebase, read_file, list_directory, run_command],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
