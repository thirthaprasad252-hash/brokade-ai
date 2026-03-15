import streamlit as st
import tempfile, os, platform
from dotenv import load_dotenv
from rag.loader import load_file, chunk_text
from rag.embedder import build_index, embed_query
from rag.retriever import retrieve
from rag.generator import generate_answer

load_dotenv()

if platform.system() != "Windows":
    tempfile.tempdir = "/tmp"

st.set_page_config(page_title="BROKADE AI", page_icon="😎", layout="wide")

SUPPORTED_TYPES = ["pdf", "docx", "txt"]

defaults = {"messages": [], "index": None, "chunks": [], "processed_files": []}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.title(" BROKADE AI")
    st.caption("I'm your Personal Document Analyser ")
    st.divider()
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files
                     if f.name not in st.session_state.processed_files]
        if new_files:
            progress = st.progress(0, text="Processing files...")
            new_chunks = []
            for i, file in enumerate(new_files):
                ext = file.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                try:
                    text = load_file(tmp_path, ext)
                    chunks = chunk_text(text)
                    new_chunks.extend(chunks)
                    st.session_state.processed_files.append(file.name)
                except Exception as e:
                    st.error(f"Failed: {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)
                progress.progress((i + 1) / len(new_files),
                                  text=f"Processed {i+1}/{len(new_files)} files...")
            st.session_state.chunks.extend(new_chunks)
            st.session_state.index, _ = build_index(st.session_state.chunks)
            progress.empty()
            st.success(f"✅ {len(new_files)} file(s) ready!")

    if st.session_state.processed_files:
        st.divider()
        st.markdown("**📋 Indexed files:**")
        for fname in st.session_state.processed_files:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📄", "DOCX": "📝", "TXT": "📃"}.get(ext, "📁")
            st.markdown(f"{icon} `{fname}`")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"💾 {len(st.session_state.chunks)} chunks")
        with col2:
            st.caption(f"💬 {len(st.session_state.messages)} messages")
        if st.button("🗑️ Clear All", use_container_width=True):
            for key, val in defaults.items():
                st.session_state[key] = [] if key != "index" else None
            st.rerun()

st.title("Hey! Upload Your Documents To Chat📄")

if not st.session_state.processed_files:
    st.info("👈 Upload a PDF, DOCX, or TXT from the sidebar to get started!")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching & thinking..."):
            query_emb = embed_query(prompt)
            context = retrieve(query_emb, st.session_state.index,
                               st.session_state.chunks, top_k=5)
            history = st.session_state.messages[:-1]
            answer = generate_answer(prompt, context, history)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})