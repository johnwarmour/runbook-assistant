import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_DIR = Path("./runbook_index")
MANIFEST_PATH = INDEX_DIR / "manifest.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

SYSTEM_PROMPT = """\
You are an expert incident responder with deep knowledge of the runbooks provided.
When answering, ground your response in the retrieved runbook content.
Be direct and actionable — the user may be in the middle of an incident.
If the runbooks don't cover the question, say so clearly rather than guessing.
Always cite which runbook(s) you are drawing from."""


# ── Index helpers ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_index() -> Optional[FAISS]:
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
    return None


def save_index(index: FAISS):
    INDEX_DIR.mkdir(exist_ok=True)
    index.save_local(str(INDEX_DIR))


def load_manifest() -> list[str]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return []


def save_manifest(files: list[str]):
    INDEX_DIR.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(files))


def index_file(uploaded_file) -> tuple[bool, str]:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        if not chunks:
            return False, "No content extracted."

        embeddings = get_embeddings()
        new_index = FAISS.from_documents(chunks, embeddings)

        existing = load_index()
        if existing:
            existing.merge_from(new_index)
            save_index(existing)
        else:
            save_index(new_index)

        manifest = load_manifest()
        if uploaded_file.name not in manifest:
            manifest.append(uploaded_file.name)
            save_manifest(manifest)

        return True, f"Indexed {len(chunks)} chunks."

    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def retrieve(query: str, k: int = TOP_K) -> list:
    index = load_index()
    if not index:
        return []
    return index.similarity_search(query, k=k)


# ── Generation ────────────────────────────────────────────────────────────────

def build_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def stream_response(query: str, docs: list, history: list):
    client = Anthropic()
    context = build_context(docs)

    messages = []
    for turn in history[:-1]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Relevant runbook content:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
        ),
    })

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Runbook Assistant", page_icon="📋", layout="wide")
st.title("Runbook Assistant")
st.caption("Ask questions about your runbooks and playbooks during incidents.")

# Sidebar — file management
with st.sidebar:
    st.subheader("Runbooks")

    uploaded = st.file_uploader(
        "Upload runbooks",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Embed", type="primary", disabled=not uploaded):
        for f in uploaded:
            with st.spinner(f"Indexing {f.name}..."):
                ok, msg = index_file(f)
            if ok:
                st.toast(f"{f.name}: {msg}", icon="✅")
            else:
                st.error(f"{f.name}: {msg}", icon="🚫")
        st.rerun()

    st.divider()

    manifest = load_manifest()
    if manifest:
        st.caption("Indexed runbooks:")
        for name in manifest:
            st.markdown(f"- `{name}`")
    else:
        st.info("No runbooks indexed yet.", icon="ℹ️")

    st.divider()

    if st.button("Clear Index", type="secondary"):
        for f in INDEX_DIR.glob("*"):
            f.unlink()
        st.session_state.pop("messages", None)
        st.toast("Index cleared.", icon="✅")
        st.rerun()

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}**")
                    st.markdown(src["content"])
                    st.divider()

if query := st.chat_input("Ask about a runbook or describe an incident symptom..."):
    if not load_manifest():
        st.warning("Upload and embed at least one runbook first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    docs = retrieve(query)
    sources = [
        {"source": d.metadata.get("source", "Unknown"), "content": d.page_content}
        for d in docs
    ]

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in stream_response(query, docs, st.session_state.messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.caption(f"**{src['source']}**")
                    st.markdown(src["content"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
