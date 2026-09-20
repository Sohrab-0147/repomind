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
- search_codebase: semantic search (use ONCE per question)
- read_file, list_directory, run_command

RULES:
1. Search codebase ONCE with a clear query.
2. If results are relevant, ANSWER IMMEDIATELY.
3. Do NOT search multiple times with similar queries.
4. Cite file names and line numbers.
5. If you cannot find the answer, say so."""


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
