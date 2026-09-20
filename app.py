import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from repomind.agent.orchestrator import handle_query_stream
from repomind.context.indexers.semantic_qdrant import index_codebase
from repomind.config import load_config

st.set_page_config(
    page_title="RepoMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@keyframes rm-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes rm-dot-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    color: #111827;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: #fbfbfd; }

.block-container {
    max-width: 1080px;
    padding-top: 2.5rem;
    padding-bottom: 6rem;
}

.rm-hero {
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 2rem;
    animation: rm-fade-in 0.5s ease-out;
}
.rm-logo {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4338ca;
    margin-bottom: 1.25rem;
}
.rm-logo-dot {
    width: 8px; height: 8px;
    background: #4f46e5;
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(79,70,229,0.18);
    animation: rm-dot-pulse 2s ease-in-out infinite;
}
.rm-title {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.05;
    margin: 0 0 0.9rem 0;
    color: #0f172a;
}
.rm-subtitle {
    font-size: 1.08rem;
    color: #374151;
    max-width: 660px;
    line-height: 1.65;
    margin: 0;
    font-weight: 400;
}
.rm-features {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.rm-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.79rem;
    font-weight: 600;
    color: #1f2937;
    background: #ffffff;
    border: 1px solid #d1d5db;
    padding: 0.32rem 0.7rem;
    border-radius: 999px;
}
.rm-chip::before {
    content: "";
    width: 6px; height: 6px;
    background: #059669;
    border-radius: 50%;
}

.rm-empty-title {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 2rem 0 1rem 0;
}

[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    animation: rm-fade-in 0.3s ease-out;
    color: #111827;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #111827 !important;
}

.stMarkdown pre {
    background: #0f172a !important;
    border-radius: 10px;
    border: none;
    padding: 1rem 1.15rem;
}
.stMarkdown pre code { color: #e5e7eb !important; }
.stMarkdown code {
    background: #eef2ff;
    color: #3730a3;
    padding: 0.12rem 0.4rem;
    border-radius: 5px;
    font-weight: 500;
}

.rm-suggest .stButton > button {
    width: 100%;
    text-align: left;
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    font-size: 0.92rem;
    font-weight: 500;
    color: #1f2937;
    transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.rm-suggest .stButton > button:hover {
    border-color: #4f46e5;
    color: #3730a3;
    box-shadow: 0 6px 20px -8px rgba(79,70,229,0.28);
}

[data-testid="stChatInput"] {
    border: 1px solid #d1d5db;
    border-radius: 14px;
    box-shadow: 0 8px 32px -12px rgba(15,23,42,0.15);
    background: #ffffff;
}
[data-testid="stChatInput"] textarea {
    color: #111827 !important;
    font-size: 0.98rem;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: #1f2937 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
}

.rm-footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.82rem;
    padding: 3rem 0 1rem 0;
    border-top: 1px solid #e5e7eb;
    margin-top: 3rem;
}
.rm-footer a {
    color: #4f46e5;
    text-decoration: none;
    font-weight: 500;
}
.rm-footer a:hover { text-decoration: underline; }

.stButton > button {
    color: #1f2937;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_index(repo_path: str):
    index_codebase(repo_path)
    return True


def _default_repo_path() -> str:
    cwd = Path.cwd()
    if (cwd / "repomind").is_dir():
        return str(cwd)
    here = Path(__file__).resolve().parent
    if (here / "repomind").is_dir():
        return str(here)
    return str(cwd)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "repo_path" not in st.session_state:
    st.session_state.repo_path = _default_repo_path()
if "pending" not in st.session_state:
    st.session_state.pending = None


cfg = load_config()
llm_model = cfg["llm"]["model"]
embed_model = cfg["embeddings"]["model"]
collection = cfg["vector_store"]["collection_name"]


st.markdown("""
<div class="rm-hero">
  <div class="rm-logo"><span class="rm-logo-dot"></span> RepoMind</div>
  <h1 class="rm-title">Ask your codebase anything.</h1>
  <p class="rm-subtitle">
    Semantic search over any repository, powered by Tree-sitter AST parsing,
    local embeddings, and an autonomous agent that cites every answer.
  </p>
  <div class="rm-features">
    <span class="rm-chip">Tree-sitter AST</span>
    <span class="rm-chip">Local BGE embeddings</span>
    <span class="rm-chip">Qdrant vector search</span>
    <span class="rm-chip">Sandboxed tools</span>
    <span class="rm-chip">Persistent memory</span>
  </div>
</div>
""", unsafe_allow_html=True)


with st.expander("Runtime info", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM", llm_model.split("/")[-1][:18])
    c2.metric("Embedder", embed_model.split("/")[-1][:18])
    c3.metric("Vector Store", "Qdrant")
    c4.metric("Session", st.session_state.session_id[:8])

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Reindex codebase", use_container_width=True):
            st.cache_resource.clear()
            with st.spinner("Reindexing..."):
                index_codebase(st.session_state.repo_path, force_reindex=True)
            st.success("Reindexed")
    with b2:
        if st.button("New chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()


with st.spinner("Indexing codebase..."):
    try:
        load_index(st.session_state.repo_path)
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        st.stop()


if not st.session_state.messages:
    st.markdown('<div class="rm-empty-title">Suggested queries</div>', unsafe_allow_html=True)
    examples = [
        ("Explain the indexing pipeline", "How does the code parser split files?"),
        ("Where is the sandbox enforced?", "Show me the path validation code"),
        ("How does the config loader work?", "What does load_config do?"),
        ("What tools does the agent have?", "List the agent tools"),
    ]
    cols = st.columns(2)
    for i, (label, query) in enumerate(examples):
        with cols[i % 2]:
            st.markdown('<div class="rm-suggest">', unsafe_allow_html=True)
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending = query
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


TOOL_META = {
    "search_codebase": ("🔍", "Searching codebase"),
    "read_file": ("📄", "Reading file"),
    "list_directory": ("📁", "Listing directory"),
    "run_command": ("⚡", "Running command"),
    "run_in_directory": ("⚡", "Running command"),
}


def _run(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        status = st.empty()
        answer = st.empty()
        buffer = ""

        try:
            for ev in handle_query_stream(prompt, st.session_state.session_id):
                if ev["type"] == "tool":
                    icon, label = TOOL_META.get(
                        ev["name"], ("⚙️", f"Calling {ev['name']}")
                    )
                    status.markdown(
                        f"<div style='color:#4b5563;font-size:0.86rem;"
                        f"padding:0.25rem 0;font-weight:500;'>"
                        f"{icon} {label}…</div>",
                        unsafe_allow_html=True,
                    )
                elif ev["type"] == "tool_done":
                    status.empty()
                elif ev["type"] == "text":
                    buffer += ev["content"]
                    status.empty()
                    answer.markdown(buffer)

            response = buffer or "No response generated."
        except Exception as e:
            response = f"Error: {e}"
            answer.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if st.session_state.pending:
    p = st.session_state.pending
    st.session_state.pending = None
    _run(p)


if prompt := st.chat_input("Ask a question about the codebase..."):
    _run(prompt)


st.markdown(
    '<div class="rm-footer">RepoMind · '
    '<a href="https://github.com/Sohrab-0147/repomind">GitHub</a></div>',
    unsafe_allow_html=True,
)
