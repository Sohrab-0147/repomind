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
    initial_sidebar_state="expanded",
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
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #0f172a;
}

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
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 1rem;
}
.rm-logo-dot {
    width: 8px; height: 8px;
    background: #6366f1;
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.15);
    animation: rm-dot-pulse 2s ease-in-out infinite;
}
.rm-title {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.05;
    margin: 0 0 0.75rem 0;
    color: #0f172a;
}
.rm-subtitle {
    font-size: 1.1rem;
    color: #64748b;
    max-width: 640px;
    line-height: 1.6;
    margin: 0;
}
.rm-features {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.25rem;
}
.rm-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: #475569;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    transition: border-color 0.15s ease, color 0.15s ease;
}
.rm-chip:hover {
    border-color: #6366f1;
    color: #4338ca;
}
.rm-chip::before {
    content: "";
    width: 6px; height: 6px;
    background: #10b981;
    border-radius: 50%;
}

.rm-empty-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 2rem 0 0.9rem 0;
}

[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e9edf2;
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    animation: rm-fade-in 0.3s ease-out;
}

.stMarkdown pre {
    background: #0f172a !important;
    border-radius: 10px;
    border: none;
    padding: 1rem 1.15rem;
}
.stMarkdown code {
    background: #f1f5f9;
    color: #4338ca;
    padding: 0.12rem 0.4rem;
    border-radius: 5px;
}

[data-testid="stSidebar"] {
    background: #fafbfc;
    border-right: 1px solid #e9edf2;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.rm-status {
    background: #ffffff;
    border: 1px solid #e9edf2;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
}
.rm-status-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.28rem 0;
}
.rm-status-label { color: #64748b; }
.rm-status-value {
    color: #0f172a;
    font-weight: 600;
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 0.78rem;
}

.rm-suggest .stButton > button {
    width: 100%;
    text-align: left;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    font-size: 0.92rem;
    font-weight: 500;
    color: #334155;
    transition: border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
}
.rm-suggest .stButton > button:hover {
    border-color: #6366f1;
    color: #4338ca;
    box-shadow: 0 4px 16px -8px rgba(99,102,241,0.25);
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    font-size: 0.85rem;
    color: #334155;
    padding: 0.55rem;
    transition: border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #6366f1;
    color: #4338ca;
}

[data-testid="stChatInput"] {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 8px 32px -12px rgba(15,23,42,0.12);
    background: #ffffff;
}

.rm-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    padding: 3rem 0 1rem 0;
    border-top: 1px solid #f1f5f9;
    margin-top: 3rem;
}
.rm-footer a {
    color: #6366f1;
    text-decoration: none;
}
.rm-footer a:hover {
    text-decoration: underline;
}

/* Force sidebar toggle visible */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #6366f1 !important;
    z-index: 999999 !important;
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


with st.sidebar:
    st.markdown("#### Settings")

    st.session_state.repo_path = st.text_input(
        "Repository",
        value=st.session_state.repo_path,
        label_visibility="collapsed",
        placeholder="Path to repository",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reindex", use_container_width=True):
            st.cache_resource.clear()
            with st.spinner("Reindexing..."):
                index_codebase(st.session_state.repo_path, force_reindex=True)
            st.toast("Index rebuilt", icon="✓")
    with c2:
        if st.button("New chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    st.markdown("#### Runtime")
    st.markdown(f"""
<div class="rm-status">
  <div class="rm-status-row"><span class="rm-status-label">LLM</span><span class="rm-status-value">{llm_model.split('/')[-1]}</span></div>
  <div class="rm-status-row"><span class="rm-status-label">Embedder</span><span class="rm-status-value">{embed_model.split('/')[-1]}</span></div>
  <div class="rm-status-row"><span class="rm-status-label">Vector store</span><span class="rm-status-value">Qdrant</span></div>
  <div class="rm-status-row"><span class="rm-status-label">Collection</span><span class="rm-status-value">{collection}</span></div>
  <div class="rm-status-row"><span class="rm-status-label">Session</span><span class="rm-status-value">{st.session_state.session_id[:8]}</span></div>
</div>
""", unsafe_allow_html=True)

    st.caption("Filesystem and shell tools are sandboxed to the workspace. Dangerous commands are blocked.")


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


def _run(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        try:
            response = st.write_stream(
                handle_query_stream(prompt, st.session_state.session_id)
            )
        except Exception as e:
            response = f"Error: {e}"
            st.markdown(response)
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
