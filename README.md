# RepoMind

RAG-powered code assistant — ask questions about any codebase and get grounded answers with file citations.

## What It Does
- Parses source files into function/class-level chunks using **Tree-sitter AST**
- Embeds chunks locally with **BAAI/bge-small-en-v1.5** (384-dim, no API cost)
- Stores vectors in **Qdrant Cloud**
- Answers questions using a **LangChain 1.0 agent** with tool-calling
- Tools: `search_codebase`, `read_file`, `list_directory`, `run_command` (sandboxed)
- Persistent conversation memory via **SQLite checkpointer**

## Tech Stack
| Layer | Tech |
|-------|------|
| LLM | Groq (openai/gpt-oss-120b) |
| Embeddings | HuggingFace sentence-transformers (local) |
| Vector store | Qdrant Cloud |
| Agent | LangChain 1.0 + LangGraph |
| UI | Streamlit |
| Parser | Tree-sitter |

## Safety
- Workspace sandbox - file tools reject any path outside the project
- Blocked commands - rm -rf /, mkfs, fork bombs are rejected
- 30s timeout on all shell commands

## Run
poetry install
poetry run repomind

Or web UI:
poetry run streamlit run app.py

## Commands
- /ask <question> - ask about the codebase
- /new_session - start fresh conversation
- /switch <id> - resume a past session
- /session - show current session id
- /exit - quit
