# 🧠 DocMind AI

> A RAG-powered document intelligence assistant — upload any PDF or text file and chat with it using Gemini 2.5 Flash + ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat-square&logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-8A2BE2?style=flat-square)

---

## ✨ Features

- 📄 **PDF & TXT/MD support** — Upload any document and start chatting instantly
- 🔢 **Gemini Embeddings** — Uses `gemini-embedding-001` via the new `google-genai` SDK with correct task-type routing (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`)
- 💾 **ChromaDB persistence** — Vector store persists between sessions; auto-loads on restart
- ⚡ **Gemini 2.5 Flash** — Fast, accurate LLM responses grounded in your document
- 📎 **Source citations** — Every answer shows which chunks (with page numbers & similarity scores) were used
- ⚙️ **Tunable RAG params** — Chunk size, overlap, Top-K, and temperature all adjustable from the sidebar
- 🎨 **Dark-themed UI** — Clean, minimal Streamlit interface with JetBrains Mono + Syne fonts

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/docmind-ai.git
cd docmind-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Open `docmind.py` and replace the placeholder:

```python
GOOGLE_API_KEY = "your-google-api-key-here"
```

Or use an environment variable (recommended):

```bash
export GOOGLE_API_KEY="your-key-here"
```

> Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run

```bash
streamlit run docmind.py
```

---

## 🏗️ Architecture

```
User uploads PDF/TXT
        ↓
PyMuPDF / TextLoader  →  RecursiveCharacterTextSplitter
        ↓
GeminiEmbeddings (gemini-embedding-001, RETRIEVAL_DOCUMENT)
        ↓
ChromaDB (persisted at ./chroma_store)
        ↓
User asks a question
        ↓
GeminiEmbeddings (gemini-embedding-001, RETRIEVAL_QUERY)
        ↓
similarity_search_with_score (Top-K chunks)
        ↓
Gemini 2.5 Flash + System Prompt
        ↓
Answer + Source Citations
```

---

## ⚙️ RAG Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Chunk size | 600 | Token size per chunk |
| Chunk overlap | 80 | Overlap between adjacent chunks |
| Top-K | 3 | Chunks retrieved per query |
| Temperature | 0.2 | LLM response creativity |

---

## 📦 Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| LLM | `langchain-google-genai` → Gemini 2.5 Flash |
| Embeddings | `google-genai` SDK → `gemini-embedding-001` |
| Vector Store | ChromaDB (`langchain-chroma`) |
| PDF Parsing | PyMuPDF (`langchain-community`) |
| Text Splitting | `langchain-text-splitters` |

---

## 📁 Project Structure

```
docmind-ai/
├── docmind.py          # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

> `chroma_store/` is created at runtime and excluded via `.gitignore`.

---

## 🔐 Security Note

Do **not** commit your API key. Use environment variables or [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) when deploying.

---

## 📄 License

MIT — free to use and modify.
