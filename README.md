# AI Risk Engine

An AI-powered financial risk analysis tool. Upload a financial PDF and ask questions — get a professional risk assessment with a score and risk level.

![AI Risk Engine](https://img.shields.io/badge/FastAPI-0.136-green) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Ollama-orange)

---

## Features

- Upload any financial PDF (annual reports, balance sheets, etc.)
- Ask natural language questions about risk
- Get a risk score (0–100) and risk level (Low / Moderate / High / Critical)
- Animated dashboard UI with history
- Runs fully local (Ollama) or in the cloud (Groq)

---

## Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/VamsiNamballa/ai-risk-engine-PavanNamballa
cd ai-risk-engine
```

### 2. Create a virtual environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model
- Download Ollama from https://ollama.com
- Then run:
```bash
ollama pull llama3.2:1b
ollama serve
```

### 5. Start the server
```bash
uvicorn app:app --reload
```

### 6. Open in browser
```
http://127.0.0.1:8000
```

---

## Cloud Setup (Groq — Free API)

To run without Ollama (e.g. on a server or for sharing):

1. Get a free API key at https://console.groq.com
2. Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
3. Start the server — it will automatically use Groq instead of Ollama.

---

## Docker

```bash
docker build -t ai-risk-engine .
docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway ai-risk-engine
```

---

## How It Works

1. **Upload** a PDF → text is extracted and split into chunks
2. Chunks are embedded and stored in **ChromaDB** (vector store)
3. Your **question** is used to search for the most relevant chunks
4. The chunks + question are sent to **LLaMA 3** (via Groq or Ollama)
5. The model returns a structured JSON with answer, risk score, and risk level

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM (local) | Ollama + LLaMA 3.2 1B |
| LLM (cloud) | Groq + LLaMA 3 8B |
| Frontend | Vanilla HTML/CSS/JS |
