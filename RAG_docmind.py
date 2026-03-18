import os
import time
import tempfile
import hashlib
import streamlit as st
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
#MainMenu, footer, { visibility: hidden; }
header {background: transparent;}            
.stApp { background-color: #0d0f14; }
[data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }
[data-testid="stChatMessage"] { background-color: #151820 !important; border: 1px solid #1e2130 !important; border-radius: 12px !important; margin-bottom: 8px !important; }
[data-testid="stChatInput"] textarea { background-color: #151820 !important; border: 1px solid #2a2f45 !important; border-radius: 10px !important; color: #e8eaf0 !important; font-family: 'Syne', sans-serif !important; }
.stButton > button { background-color: #1a1f30 !important; border: 1px solid #2a2f45 !important; color: #a0c4ff !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; transition: all 0.2s ease !important; }
.stButton > button:hover { border-color: #a0c4ff !important; background-color: #1e2540 !important; }
[data-testid="stMetric"] { background-color: #151820; border: 1px solid #1e2130; border-radius: 10px; padding: 10px 14px !important; }
.citation-box { background-color: #0e1420; border-left: 3px solid #a0c4ff; border-radius: 0 8px 8px 0; padding: 8px 14px; margin-top: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #7a8ab0; }
[data-testid="stFileUploader"] { background-color: #151820 !important; border: 1px dashed #2a2f45 !important; border-radius: 12px !important; }
[data-testid="stStatus"] { background-color: #151820 !important; border-color: #1e2130 !important; }
hr { border-color: #1e2130 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2f45; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = "AIzaSyCQjX_fP7rUtooIZGdHdE0wLulRq2T7WX8"
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "docmind_kb"

# new google-genai SDK client
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
client = genai.Client(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are DocMind, a precise and intelligent document assistant.
You answer questions strictly based on the provided document context.

Rules:
- Use the retrieved context as primary reference, but also apply your own knowledge to answer questions thoroughly.
- If the answer isn't in the context, say: "I couldn't find that in the uploaded document."
- Always be concise, accurate, and cite which part of the document supports your answer.
- Never hallucinate or make up information not present in the context.
- Format answers clearly. Use bullet points or numbered lists when helpful."""

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Custom Embeddings using google-genai SDK directly ─────────────────────────
from langchain_core.embeddings import Embeddings
from typing import List

class GeminiEmbeddings(Embeddings):
    """Uses google-genai (new SDK) directly — stable v1 API, no v1beta routing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for text in texts:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            result.append(response.embeddings[0].values)
        return result

    def embed_query(self, text: str) -> List[float]:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

# ── Helpers ────────────────────────────────────────────────────────────────────
def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def load_and_chunk(file_path, file_type, chunk_size, chunk_overlap):
    loader = PyMuPDFLoader(file_path) if file_type == "pdf" else TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=GeminiEmbeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

def load_existing_vectorstore():
    try:
        vs = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=GeminiEmbeddings(),
            collection_name=COLLECTION_NAME
        )
        count = vs._collection.count()
        if count > 0:
            return vs, count
    except Exception:
        pass
    return None, 0

def retrieve_and_answer(query, vectorstore, top_k, temp):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    if not results:
        return "I couldn't find relevant content in the uploaded document.", []

    context_parts, citations = [], []
    for i, (doc, score) in enumerate(results):
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Chunk {i+1} | Page {page}]:\n{doc.page_content}")
        citations.append({
            "chunk": i + 1,
            "page": page,
            "score": round(float(score), 3),
            "preview": doc.page_content[:120].replace("\n", " ") + "..."
        })

    context = "\n\n---\n\n".join(context_parts)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context from document:\n\n{context}\n\n---\n\nQuestion: {query}")
    ]
    response = llm.invoke(messages)
    return response.content, citations

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.caption("RAG · Gemini 2.5 Flash · ChromaDB")
    st.divider()

    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT/MD", type=["pdf", "txt", "md"], label_visibility="collapsed")

    st.markdown("### ⚙️ Chunking")
    chunk_size = st.slider("Chunk size (tokens)", 200, 1500, 600, step=50, help="Larger = more context per chunk")
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 80, step=20, help="Overlap between chunks")

    st.markdown("### 🎯 Retrieval")
    top_k = st.slider("Top-K chunks", 1, 8, 3, help="How many chunks to retrieve per query")
    temp = st.slider("Response creativity", 0.0, 1.0, 0.2, help="Lower = more factual")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gemini API", "⚡ On")
    with col2:
        st.metric("ChromaDB", "💾 On")

    if st.session_state.doc_name:
        st.success(f"📎 {st.session_state.doc_name}")
        st.caption(f"Chunks indexed: **{st.session_state.chunk_count}**")

    st.divider()

    if st.session_state.vectorstore is None:
        existing_vs, count = load_existing_vectorstore()
        if existing_vs:
            st.session_state.vectorstore = existing_vs
            st.session_state.chunk_count = count
            if st.session_state.doc_name is None:
                st.session_state.doc_name = "Previously indexed doc"

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("💣 Wipe KB", use_container_width=True):
            import shutil
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            st.session_state.vectorstore = None
            st.session_state.doc_name = None
            st.session_state.doc_hash = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

# ── Process upload ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    fhash = file_hash(file_bytes)

    if fhash != st.session_state.doc_hash:
        file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "txt"
        with st.sidebar:
            with st.status("🔬 Indexing document...", expanded=True) as status:
                st.write("📖 Loading and parsing...")
                suffix = ".pdf" if file_type == "pdf" else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                chunks = load_and_chunk(tmp_path, file_type, chunk_size, chunk_overlap)
                st.write(f"✂️ Split into {len(chunks)} chunks")
                st.write("🔢 Generating embeddings...")
                vs = build_vectorstore(chunks)
                os.unlink(tmp_path)

                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_hash = fhash
                st.session_state.chunk_count = len(chunks)
                st.session_state.messages = []
                status.update(label=f"✅ Indexed {len(chunks)} chunks!", state="complete")
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("## Where's the knowledge? 📡")
st.caption("DocMind AI · Ask anything about your uploaded document")

if st.session_state.vectorstore is None:
    st.info("⬅️ Upload a PDF or TXT file in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        role, icon = ("user", "🧑") if msg["role"] == "user" else ("assistant", "🧠")
        with st.chat_message(role, avatar=icon):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Source chunks", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(
                            f'<div class="citation-box">Chunk {c["chunk"]} · Page {c["page"]} · similarity: {c["score"]}<br>'
                            f'<span style="color:#9ab">{c["preview"]}</span></div>',
                            unsafe_allow_html=True
                        )

    if prompt := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            with st.status("🔍 Searching knowledge base...") as status:
                st.write("Embedding your query...")
                time.sleep(0.3)
                st.write(f"Retrieving top-{top_k} chunks...")
                raw_answer, citations = retrieve_and_answer(prompt, st.session_state.vectorstore, top_k, temp)
                answer = extract_text(raw_answer)
                status.update(label="✅ Found relevant context", state="complete")

            msg_box = st.empty()
            typed = ""
            for ch in answer:
                typed += ch
                msg_box.markdown(typed + "▌")
                time.sleep(0.008)
            msg_box.markdown(typed)

            if citations:
                with st.expander("📎 Source chunks", expanded=False):
                    for c in citations:
                        st.markdown(
                            f'<div class="citation-box">Chunk {c["chunk"]} · Page {c["page"]} · similarity: {c["score"]}<br>'
                            f'<span style="color:#9ab">{c["preview"]}</span></div>',
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})