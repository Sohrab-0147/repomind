import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "GITHUB_TOKEN",
                "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from repomind.agent.orchestrator import handle_query_stream
from repomind.context.indexers.semantic_qdrant import index_codebase
from repomind.config import load_config

st.set_page_config(
    page_title="RepoMind — Ask your codebase anything",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink:        #0b1020;
    --ink-2:      #1f2937;
    --ink-3:      #4b5563;
    --muted:      #6b7280;
    --line:       #e6e8ee;
    --line-2:     #eef0f5;
    --bg:         #fafbfd;
    --card:       #ffffff;
    --brand-1:    #4f46e5;
    --brand-2:    #7c3aed;
    --brand-3:    #ec4899;
    --brand-soft: #eef2ff;
}

@keyframes rm-fade-up {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes rm-gradient-drift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes rm-dot {
    0%, 100% { transform: scale(1);   opacity: 1;   }
    50%      { transform: scale(1.3); opacity: 0.55; }
}
@keyframes rm-ring {
    0%   { box-shadow: 0 0 0 0   rgba(79,70,229,0.35); }
    70%  { box-shadow: 0 0 0 10px rgba(79,70,229,0);   }
    100% { box-shadow: 0 0 0 0   rgba(79,70,229,0);   }
}

#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg); }

/* faint radial gradient in the top-right corner — restrained, not loud */
.stApp::before {
    content: "";
    position: fixed;
    top: -280px;
    right: -280px;
    width: 720px;
    height: 720px;
    background: radial-gradient(circle, rgba(124,58,237,0.09), rgba(79,70,229,0.03) 45%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.75rem;
    padding-bottom: 6rem;
    position: relative;
    z-index: 1;
}

/* ── HERO ────────────────────────────────────────────── */
.rm-hero { padding: 0 0 2rem 0; margin-bottom: 1.5rem; }

.rm-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.38rem 0.85rem 0.38rem 0.55rem;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--ink-3);
    margin-bottom: 1.4rem;
    animation: rm-fade-up 0.5s ease-out;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.rm-eyebrow-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
    animation: rm-dot 2.4s ease-in-out infinite;
}
.rm-eyebrow strong { color: var(--ink); font-weight: 700; }

.rm-title {
    font-size: 3.05rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.03;
    margin: 0 0 1rem 0;
    color: var(--ink);
    animation: rm-fade-up 0.6s ease-out;
}
.rm-title .grad {
    background: linear-gradient(135deg, var(--brand-1) 0%, var(--brand-2) 45%, var(--brand-3) 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: rm-gradient-drift 9s ease infinite;
}

.rm-subtitle {
    font-size: 1.1rem;
    color: var(--ink-3);
    max-width: 700px;
    line-height: 1.65;
    margin: 0 0 1.75rem 0;
    animation: rm-fade-up 0.7s ease-out;
}

.rm-chips {
    display: flex; gap: 0.45rem; flex-wrap: wrap;
    margin-bottom: 0.5rem;
    animation: rm-fade-up 0.85s ease-out;
}
.rm-chip {
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--ink-2);
    background: #ffffff;
    border: 1px solid var(--line);
    padding: 0.32rem 0.72rem;
    border-radius: 999px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.02);
}
.rm-chip::before {
    content: "";
    display: inline-block;
    width: 6px; height: 6px;
    margin-right: 0.4rem;
    background: #10b981;
    border-radius: 50%;
    vertical-align: middle;
}

/* ── STEPS ───────────────────────────────────────────── */
.rm-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-top: 2.25rem;
    animation: rm-fade-up 0.95s ease-out;
}
.rm-step {
    position: relative;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.1rem 1.15rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.rm-step:hover {
    border-color: #c7cbe0;
    box-shadow: 0 6px 20px -12px rgba(15,23,42,0.12);
}
.rm-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--brand-soft), #f5f0ff);
    color: var(--brand-1);
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}
.rm-step-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0 0 0.28rem 0;
}
.rm-step-desc {
    font-size: 0.82rem;
    color: var(--ink-3);
    line-height: 1.55;
    margin: 0;
}

/* ── SECTION HEADINGS ───────────────────────────────── */
.rm-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2.25rem 0 0.4rem 0;
}
.rm-section-sub {
    font-size: 0.92rem;
    color: var(--ink-3);
    margin: 0 0 1rem 0;
}

.rm-cat {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.83rem;
    font-weight: 700;
    color: var(--ink);
    margin: 1.25rem 0 0.55rem 0;
}
.rm-cat .ico {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 6px;
    background: var(--brand-soft);
    font-size: 0.85rem;
}

/* ── ACTION BUTTONS ─────────────────────────────────── */
.rm-action .stButton > button {
    width: 100%;
    text-align: left;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 0.8rem 0.95rem;
    font-size: 0.86rem;
    font-weight: 500;
    color: var(--ink-2);
    line-height: 1.4;
    transition: border-color 0.15s ease, color 0.15s ease,
                box-shadow 0.15s ease, transform 0.15s ease;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.rm-action .stButton > button:hover {
    border-color: var(--brand-1);
    color: var(--brand-1);
    box-shadow: 0 8px 22px -12px rgba(79,70,229,0.4);
    transform: translateY(-1px);
}

/* ── CHAT ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    animation: rm-fade-up 0.35s ease-out;
    color: var(--ink);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th { color: var(--ink) !important; }

.stMarkdown pre {
    background: #0b1020 !important;
    border-radius: 11px;
    border: 1px solid #1f2937;
    padding: 1rem 1.1rem;
}
.stMarkdown pre code { color: #e5e7eb !important; font-size: 0.85rem; }
.stMarkdown code {
    background: var(--brand-soft);
    color: var(--brand-1);
    padding: 0.12rem 0.4rem;
    border-radius: 5px;
    font-weight: 500;
    font-size: 0.88em;
}

/* ── CHAT INPUT ─────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--card);
    border: 1.5px solid var(--line) !important;
    border-radius: 14px;
    box-shadow: 0 12px 34px -16px rgba(15,23,42,0.18);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--brand-1) !important;
    box-shadow: 0 12px 34px -12px rgba(79,70,229,0.3);
}
[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    font-size: 1rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9aa1ae !important; }

/* ── EXPANDERS ──────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--line) !important;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
[data-testid="stExpander"] summary {
    color: var(--ink-2) !important;
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0.7rem 1rem;
}

[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-weight: 700;
    font-size: 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--ink-3) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── FOOTER ─────────────────────────────────────────── */
.rm-footer {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.25rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--line-2);
    font-size: 0.82rem;
    color: var(--muted);
}
.rm-footer a {
    color: var(--ink-3);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s ease;
}
.rm-footer a:hover { color: var(--brand-1); }

.stButton > button { color: var(--ink-2); }
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


# ─────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="rm-hero">
  <div class="rm-eyebrow">
    <span class="rm-eyebrow-dot"></span>
    <strong>RepoMind</strong>&nbsp;·&nbsp;Agentic code intelligence
  </div>
  <h1 class="rm-title">Ask your codebase<br/><span class="grad">anything.</span></h1>
  <p class="rm-subtitle">
    Semantic search over any repository — powered by Tree-sitter AST parsing,
    local BGE embeddings, Qdrant vector search, and a LangChain agent with
    sandboxed tools and GitHub integration.
  </p>
  <div class="rm-chips">
    <span class="rm-chip">Tree-sitter AST</span>
    <span class="rm-chip">Local embeddings</span>
    <span class="rm-chip">Qdrant Cloud</span>
    <span class="rm-chip">MCP · GitHub</span>
    <span class="rm-chip">Sandboxed tools</span>
    <span class="rm-chip">Persistent memory</span>
  </div>
  <div class="rm-steps">
    <div class="rm-step">
      <div class="rm-step-num">1</div>
      <div class="rm-step-title">Pick a question</div>
      <p class="rm-step-desc">Click any of the suggestions below, or type your own question in the chat bar.</p>
    </div>
    <div class="rm-step">
      <div class="rm-step-num">2</div>
      <div class="rm-step-title">The agent searches</div>
      <p class="rm-step-desc">Watch live tool indicators — the agent searches code, reads files, and queries GitHub.</p>
    </div>
    <div class="rm-step">
      <div class="rm-step-num">3</div>
      <div class="rm-step-title">Get a cited answer</div>
      <p class="rm-step-desc">Every answer cites the exact file, function, and line numbers it came from.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# RUNTIME + BEGINNER GUIDE
# ─────────────────────────────────────────────────────
with st.expander("⚙️  Runtime info & controls", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM", llm_model.split("/")[-1][:22])
    c2.metric("Embedder", embed_model.split("/")[-1][:22])
    c3.metric("Vector store", "Qdrant")
    c4.metric("Session", st.session_state.session_id[:8])

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔄  Reindex codebase", use_container_width=True):
            st.cache_resource.clear()
            with st.spinner("Reindexing..."):
                index_codebase(st.session_state.repo_path, force_reindex=True)
            st.success("Reindexed")
    with b2:
        if st.button("✨  New chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    with b3:
        st.caption(f"Repo · `{Path(st.session_state.repo_path).name}`")

with st.expander("❓  New here? Start with this guide", expanded=False):
    st.markdown("""
**What is RepoMind?**

It's a code assistant. You ask a question about a codebase in plain English,
and it answers with **file paths and line numbers** so you can verify the answer.

**How do I use it?**

1. **Scroll down** — pick one of the grouped questions, or type your own at the bottom.
2. **Watch the tool indicators** — `🔍 Searching codebase…`, `📄 Reading file…`, `⚡ Running command…`.
3. **Read the answer** — every claim is backed by a real file and line number.
4. **Ask a follow-up** — the agent remembers your conversation in this session.

**What can I ask?**

| Type | Example |
|------|---------|
| **Understand code** | "How does the config loader work?" |
| **Find things** | "Where is the sandbox enforced?" |
| **Trace logic** | "Explain the indexing pipeline" |
| **GitHub data** | "List recent commits on Sohrab-0147/repomind" |

**What is it safe to try?**

Everything. The agent runs in a **sandboxed workspace** — it cannot read, write,
or delete files outside this project. Dangerous shell commands are blocked in code.
    """)


# ─────────────────────────────────────────────────────
# INDEX
# ─────────────────────────────────────────────────────
with st.spinner("Indexing codebase..."):
    try:
        load_index(st.session_state.repo_path)
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        st.stop()


# ─────────────────────────────────────────────────────
# QUICK ACTIONS (shown only when chat is empty)
# ─────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="rm-section-label">Quick actions</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rm-section-sub">Pick any question below to see RepoMind in action.</p>',
        unsafe_allow_html=True,
    )

    ACTIONS = [
        ("🧠", "Understand the code", [
            ("Explain the indexing pipeline end-to-end", "How does the code parser split files?"),
            ("How does the config loader work?", "What does load_config do?"),
            ("What does the agent factory build?", "How does build_agent work?"),
        ]),
        ("🔒", "Safety & tools", [
            ("Where is the sandbox enforced?", "Show me the path validation code"),
            ("What shell commands are blocked?", "Show me the BLOCKED_COMMANDS list"),
            ("What tools does the agent have?", "List the agent tools"),
        ]),
        ("🐙", "GitHub data (MCP)", [
            ("List recent commits on Sohrab-0147/repomind", "Show me recent commits"),
            ("What files are in the repomind repo?", "List the repo files"),
            ("Search the repo for README", "Find README in the repo"),
        ]),
        ("💬", "Follow-ups & context", [
            ("What did we just talk about?", "Show session memory"),
            ("Summarize the whole project in 5 lines", "Project overview"),
            ("What would you improve next?", "Engineering roadmap"),
        ]),
    ]

    for icon, category, items in ACTIONS:
        st.markdown(
            f'<div class="rm-cat"><span class="ico">{icon}</span>{category}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, (label, query) in enumerate(items):
            with cols[i % 3]:
                st.markdown('<div class="rm-action">', unsafe_allow_html=True)
                if st.button(label, key=f"act_{icon}_{i}", use_container_width=True):
                    st.session_state.pending = query
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────
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
    "list_commits": ("🐙", "Fetching commits"),
    "list_issues": ("🐙", "Fetching issues"),
    "search_repositories": ("🐙", "Searching GitHub"),
    "get_file_contents": ("🐙", "Reading GitHub file"),
    "get_pull_request": ("🐙", "Fetching pull request"),
    "create_issue": ("🐙", "Creating issue"),
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
                        f"<div style='display:inline-flex;align-items:center;gap:0.5rem;"
                        f"background:#eef2ff;border:1px solid #e0e7ff;color:#4338ca;"
                        f"font-size:0.82rem;font-weight:600;padding:0.35rem 0.75rem;"
                        f"border-radius:999px;margin:0.25rem 0;'>{icon} {label}…</div>",
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


# ─────────────────────────────────────────────────────
# PENDING PROMPT (from a quick-action click)
# ─────────────────────────────────────────────────────
if st.session_state.pending:
    p = st.session_state.pending
    st.session_state.pending = None
    _run(p)


# ─────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the codebase…"):
    _run(prompt)


# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────
st.markdown(
    '<div class="rm-footer">'
    '<span>RepoMind · Built with Streamlit, LangChain & Qdrant</span>'
    '<a href="https://github.com/Sohrab-0147/repomind" target="_blank">GitHub</a>'
    '</div>',
    unsafe_allow_html=True,
)
