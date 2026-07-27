# 🏆 Sports Bot 2

A **RAG-powered (Retrieval-Augmented Generation)** AI chatbot that answers questions about **Cricket**, **Football**, and **Formula 1** using curated PDF knowledge bases and multi-provider LLM inference.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Core Concepts](#-core-concepts)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Knowledge Base](#-knowledge-base)
- [RAG Pipeline](#-rag-pipeline)
- [AI Inference Layer](#-ai-inference-layer)
- [Getting Started](#-getting-started)
- [Running the Server](#-running-the-server)
- [API Usage](#-api-usage)
- [Testing](#-testing)
- [Configuration](#-configuration)

---

## 🔍 Overview

Sports Bot 2 is a domain-specific AI assistant that answers questions **strictly** from a curated sports knowledge base. It uses Retrieval-Augmented Generation (RAG) to ground every answer in real data from official PDF documents — ICC cricket rankings, FIFA rankings, F1 standings, football laws, and more.

Key design principles:
- **Context-only answers** — the bot refuses to hallucinate; if data is missing from the knowledge base, it says so.
- **Multi-provider LLM with fallback** — uses Gemini and Groq as primary/secondary LLM providers with automatic failover.
- **Streaming support** — responses can be streamed token-by-token for low-latency UX.
- **Persistent vector store** — FAISS index is built once and cached on disk; subsequent startups load from cache instantly.

---

## 🧠 Core Concepts

### 1. Retrieval-Augmented Generation (RAG)

RAG is the backbone of this bot. Instead of relying solely on an LLM's parametric knowledge, RAG:

1. **Loads** raw PDF documents from the `data/` folder
2. **Splits** them into overlapping text chunks (800 chars, 100-char overlap)
3. **Embeds** each chunk into a 768-dimensional vector using `BAAI/bge-base-en-v1.5`
4. **Stores** all vectors in a FAISS HNSW index for fast similarity search
5. **At query time** — encodes the user question, finds the top-5 most similar chunks, and passes them as `CONTEXT` to the LLM

This ensures answers are always grounded in the actual data files.

### 2. Vector Similarity Search (FAISS HNSW)

The bot uses **FAISS `IndexHNSWFlat`** (Hierarchical Navigable Small World graph):
- `M=32` — neighbors per node (controls graph connectivity)
- `efConstruction=200` — build-time search depth (quality vs. speed tradeoff)
- `efSearch=64` — query-time search depth
- Results are returned with L2 distance scores

HNSW offers sub-linear search complexity — much faster than brute-force on large corpora.

### 3. Multi-Provider LLM with Retry + Fallback

The AI layer uses **LiteLLM** as a unified interface to call:
- `gemini/gemini-2.5-flash` (Google Gemini)
- `groq/llama-3.3-70b-versatile` (Groq / Meta LLaMA 3.3)

There are two ordered model lists — **`FAST`** and **`PRO`** modes — each with different provider priority. The fallback mechanism:

1. Tries the **primary** model with up to **3 retries** (1-second delay between each)
2. If all retries fail, automatically **falls back** to the secondary provider
3. The API response includes a `fallback: true/false` flag so clients can see which provider was used

### 4. Context-Constrained Prompting

The system prompt enforces strict boundaries:
- The LLM is told to use **ONLY** the provided context
- If the answer doesn't exist in the retrieved chunks, it replies with a fixed phrase
- This prevents hallucination on sports topics not covered in the knowledge base

### 5. Persistent Vectorstore with Smart Initialization

On first startup, the full RAG pipeline runs (load → split → embed → index → save). From the second startup onward, it detects the saved `vectorstore/vector.index` and `vectorstore/metadata.pkl` and loads them instantly — skipping the expensive build step.

---

## 🏗️ Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│   FastAPI App   │  (server.py → app/main.py)
│   /chat         │
│   /chat/stream  │
└────────┬────────┘
         │  1. Receive question + mode
         ▼
┌─────────────────────────────────────────────────────┐
│                  RAG Pipeline                       │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │EmbedQuery│───▶│  FAISS   │───▶│Build Context │  │
│  │(BGE v1.5)│    │ HNSW idx │    │ (top-5 docs) │  │
│  └──────────┘    └──────────┘    └──────┬───────┘  │
└────────────────────────────────────────-│-----------┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │   Prompt Builder    │
                              │ (context + question)│
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   LLM Inference     │
                              │ Retry + Fallback    │
                              │  Gemini / Groq      │
                              └──────────┬──────────┘
                                         │
                                         ▼
                                   JSON Response
                              (response / provider / model / fallback)
```

---

## 📁 Project Structure

```
Sports-Bot-2/
│
├── server.py                    # Entry point — runs uvicorn server
├── requirements.txt             # Python dependencies
│
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan hooks
│   │
│   ├── core/
│   │   └── config.py            # Model configs (FAST_MODELS, PRO_MODELS)
│   │
│   ├── routes/
│   │   └── chatbot.py           # API endpoints (/chat, /chat/stream)
│   │
│   ├── ai/
│   │   ├── llm.py               # generate_response() + stream_response()
│   │   ├── fallback.py          # Multi-model fallback orchestrator
│   │   └── retry.py             # Per-model retry logic (3 attempts)
│   │
│   ├── prompts/
│   │   ├── rag_prompt.py        # Main RAG prompt builder (build_prompt)
│   │   └── sports_prompt.py     # Sports-only system prompt constant
│   │
│   └── rag/
│       ├── startup.py           # RAGPipeline class (init + orchestration)
│       ├── loader.py            # PDFLoader — reads PDFs from data/
│       ├── splitter.py          # DocumentSplitter — recursive chunking
│       ├── embedding.py         # EmbeddingModel (BGE bge-base-en-v1.5)
│       ├── vectordb.py          # VectorDatabase (FAISS HNSW index)
│       └── retriever.py        # Retriever — query to top-k chunks
│
├── data/
│   ├── cricket/                 # 18 ICC ranking PDFs (Men + Women)
│   ├── football/                # 5 FIFA/UEFA PDFs + Laws of the Game
│   └── formula1/                # 4 F1 2026 PDFs (drivers, teams, results)
│
├── vectorstore/
│   ├── vector.index             # Persisted FAISS HNSW index
│   └── metadata.pkl             # Pickled document metadata + text
│
├── test_loader.py               # Manual test: PDF loading
├── test_splitter.py             # Manual test: text chunking
├── test_embedding.py            # Manual test: embedding pipeline
├── test_retriever.py            # Manual test: end-to-end retrieval
└── test_vectordb.py             # Manual test: vector DB save/load
```

---

## 🛠️ Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| **Web Framework** | FastAPI | REST API + streaming responses |
| **ASGI Server** | Uvicorn | Production-grade async server |
| **LLM Interface** | LiteLLM 1.76.0 | Unified API for Gemini + Groq |
| **Embeddings** | sentence-transformers | `BAAI/bge-base-en-v1.5` model |
| **Vector Search** | FAISS (faiss-cpu) | HNSW approximate nearest neighbor |
| **PDF Parsing** | PyMuPDF (fitz) | Extract text from PDF files |
| **Numerics** | NumPy | Embedding array operations |
| **Config** | python-dotenv | Load API keys from `.env` |
| **LLM Providers** | Google Gemini, Groq | Actual language model inference |

---

## 📚 Knowledge Base

### Cricket (18 PDFs)
- Men's Test / ODI / T20I — Batting, Bowling, All-Rounder, Team Rankings
- Women's ODI / T20I — Batting, Bowling, Team Rankings
- **Source**: ICC official rankings

### Football (5 PDFs)
- FIFA Coca-Cola Men's World Ranking
- FIFA Coca-Cola Women's World Ranking
- UEFA Club Coefficients / Rankings
- Laws of the Game 2025/26 (IFAB)
- VAR Protocol (IFAB)

### Formula 1 (4 PDFs)
- 2026 F1 Drivers' Standings
- 2026 F1 Race Results
- F1 Drivers 2026 (profiles)
- F1 Racing Teams 2026

---

## 🔄 RAG Pipeline

### Build Phase (first run only)

```
PDFs in data/
    │
    ▼  PDFLoader.load_documents()
Raw text documents [{source, text}]
    │
    ▼  DocumentSplitter.split_documents()
Chunks [{source, text}]  ← 800-char chunks, 100-char overlap
    │
    ▼  EmbeddingModel.create_embeddings()
Embeddings [numpy arrays, shape=(n, 768)]
    │
    ▼  EmbeddingModel.build_documents()
Embedded docs [{source, text, embedding}]
    │
    ▼  VectorDatabase.create_index() + add_documents()
FAISS HNSW Index (in memory)
    │
    ▼  VectorDatabase.save()
vectorstore/vector.index + metadata.pkl  (on disk)
```

### Query Phase (every request)

```
User query (string)
    │
    ▼  EmbeddingModel.encode()
Query embedding [numpy, shape=(768,)]
    │
    ▼  VectorDatabase.search(top_k=5)
Top-5 [{source, text, score}]
    │
    ▼  Retriever.build_context()
Context string (source + text blocks)
    │
    ▼  build_prompt(context, question)
Full LLM prompt
    │
    ▼  LLM inference (Gemini / Groq)
Answer string
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for:
  - **Google Gemini** (`GEMINI_API_KEY`)
  - **Groq** (`GROQ_API_KEY`)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd Sports-Bot-2

# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the Server

```bash
python server.py
```

The server starts at `http://127.0.0.1:8000` with **hot-reload** enabled.

On first launch, the RAG pipeline will build and index the entire PDF corpus (this may take a few minutes). Subsequent launches load the cached vectorstore instantly.

**Startup logs:**
```
============================================================
Starting SportsBot...
============================================================
============================================================
Initializing RAG Pipeline...
============================================================
Loading Existing Vector Database...
Vector Database Loaded
============================================================
SportsBot RAG Pipeline Ready
============================================================
```

---

## 🌐 API Usage

See [api_documentation.md](./api_documentation.md) for full API reference.

**Quick reference:**

```
GET /                             → Welcome message
GET /health                       → Health check
GET /chat?q=...&mode=fast         → Blocking chat response (JSON)
GET /chat/stream?q=...&mode=pro   → Streaming chat response (text/plain)
```

---

## 🧪 Testing

Individual component tests can be run manually (no test framework required):

```bash
# Test PDF loading
python test_loader.py

# Test text chunking
python test_splitter.py

# Test embedding pipeline
python test_embedding.py

# Test full retrieval pipeline
python test_retriever.py

# Test vector DB save/load cycle
python test_vectordb.py
```

---

## ⚙️ Configuration

### Chunk Settings (`app/rag/splitter.py`)

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `800` | Max characters per chunk |
| `overlap` | `100` | Characters of overlap between chunks |
| `separators` | `["\n\n", "\n", ". ", " "]` | Split hierarchy |

### Embedding Model (`app/rag/embedding.py`)

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `BAAI/bge-base-en-v1.5` | HuggingFace model |
| `batch_size` | `32` | Encoding batch size |
| `dimension` | `768` | Vector dimension |

### FAISS Index (`app/rag/vectordb.py`)

| Parameter | Default | Description |
|---|---|---|
| `dimension` | `768` | Must match embedding dimension |
| `M` | `32` | HNSW graph connections per node |
| `efConstruction` | `200` | Build-time beam width |
| `efSearch` | `64` | Query-time beam width |

### LLM Inference (`app/ai/retry.py` + `fallback.py`)

| Parameter | Default | Description |
|---|---|---|
| `temperature` | `0.4` | LLM sampling temperature |
| `max_tokens` | `250` | Max tokens in response |
| `retries` | `3` | Attempts per model before giving up |
| `retry_delay` | `1s` | Sleep between retries |

---

## 🔐 Environment Variables

| Variable | Required | Provider |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google AI Studio |
| `GROQ_API_KEY` | Yes | Groq Cloud Console |

---

## 📝 License

This project is for educational and personal use.
