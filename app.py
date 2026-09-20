import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from repomind.agent.orchestrator import handle_query_stream
from repomind.context.indexers.semantic_qdrant import index_codebase

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")

st.title("🧠 RepoMind")
st.caption("RAG-powered code assistant — ask questions about any codebase")


@st.cache_resource(show_spinner="Indexing codebase...")
def load_index(repo_path: str):
    index_codebase(repo_path)
    return True


with st.sidebar:
    st.header("⚙️ Settings")
    default_repo = str(Path.cwd())
    repo_path = st.text_input("Repository path", value=default_repo)

    if st.button("Re-index repository"):
        st.cache_resource.clear()
        with st.spinner("Re-indexing..."):
            index_codebase(repo_path, force_reindex=True)
        st.success("Re-indexed successfully")

    if st.button("New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**How to use**")
    st.markdown("- Ask a question about the code")
    st.markdown("- The agent cites files and remembers context")
    st.markdown("- Try: *'how does the config loader work?'*")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

try:
    load_index(repo_path)
except Exception as e:
    st.error(f"Indexing failed: {e}")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the codebase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                handle_query_stream(prompt, st.session_state.session_id)
            )
        except Exception as e:
            response = f"Error: {e}"
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
