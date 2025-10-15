# 🧠 MEGARAG  
### _"An overcomplicated way to RAG systems."_

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)
![Status](https://img.shields.io/badge/status-Experimental-orange)

---

## 📖 Overview

**MEGARAG** is a modular, extensible demo of a **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**.  
It aims to explore the design space of **multi-stage retrieval**, **intent classification**, and **adaptive query routing** — all with an unapologetically over-engineered architecture.

The project currently supports both **BM25** and **dense embedding** retrieval, integrates **RAPTOR-style summarization**, and exposes its logic through a simple **FastAPI backend**.  
It is designed for experimentation, benchmarking, and extensibility rather than production deployment (for now 😉).

---

## 🧩 Core Features

- 🔍 **Hybrid Retrieval Stack** — BM25 (lexical) + Dense embeddings (Chroma, etc.)  
- 🪶 **RAPTOR Summarization** — hierarchical chunking and topic clustering for efficient retrieval  
- 🧠 **Query Classifier** — detects query intent (e.g., chitchat, general knowledge, unsafe, internal knowledge)  
- 🧰 **Multi-Tenant Architecture** — each dataset has its own vector DB directory structure  
- ⚙️ **FastAPI Interface** — REST endpoints for query, ingestion, and diagnostics  
- 🧩 **LLM-Agnostic** — supports any LLM backend (tested with **Gemma 3 1B** on **Ollama**)  
- 🧪 **Modular Design** — each stage (indexing, cleaning, summarization, classification, retrieval) is isolated and reusable  
![Pipeline backend indexing](images/bck_indx_pipeline.png)


---

## 🏗️ Project Structure

```
MEGARAG/
├── back_indx/                  # Core backend logic (BM25, RAPTOR, etc.)
│   ├── bm25_pipeline.py
│   ├── raptor.py
│   ├── indx_pipeline.py
│   └── utils/
│
├── api/                        # FastAPI routes and server entry point
│   └── main.py
│
├── models/                     # Query classifier weights or LLM configs
│   └── best/
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/MelvynChemin/MEGARAG.git
cd MEGARAG
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv megarag
source megarag/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the FastAPI backend

```bash
uvicorn api.main:app --reload
```

API will be available at:

> [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧠 Example Usage

### Query an indexed dataset

```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain RAG retrieval flow"}'
```

### Ingest new documents

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"path": "data/my_docs"}'
```

---

## 🧩 Supported LLMs

MEGARAG is **LLM-agnostic** — you can connect:

* 🦙 **Ollama** (tested with Google **Gemma 3 1B**)
* 🔥 **OpenAI GPT** (via API key)
* 🧱 **Local models** (e.g., Mistral, Llama 3, DeepSeek, etc.)

Configuration can be adapted in your environment or inside `api/main.py`.

---

## ⚙️ Tech Stack

| Component     | Technology                                |
| ------------- | ----------------------------------------- |
| API Layer     | FastAPI                                   |
| Retrieval     | BM25 (rank_bm25), Chroma                  |
| Summarization | RAPTOR-style hierarchical nodes           |
| Vectorization | Sentence Transformers / OpenAI Embeddings |
| LLM Interface | Ollama / OpenAI API                       |
| Orchestration | Python 3.10+, Modular scripts             |

---

## 🧭 Roadmap

* [ ] Add front-end dashboard
* [ ] Add support for mixed-context RAG (structured + unstructured)
* [ ] Implement caching layer (FAISS/Redis)
* [ ] Integrate evaluation metrics (Recall@k, MRR, etc.)
* [ ] Add Dockerfile for easy deployment

---

## 🧑‍💻 Author

**Melvyn CHEMIN**  
Master's in AI & Data Science @ ENSIMAG  
Currently building modular multi-agent and RAG systems.  
[🔗 LinkedIn](https://www.linkedin.com/in/melvynchemin) • [🐙 GitHub](https://github.com/MelvynChemin)

---

## ⚖️ License

MIT License © 2025 Melvyn CHEMIN
