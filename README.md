
RAG AI Agent: Ask Questions About Agentic AI

A Retrieval-Augmented Generation (RAG) agent that answers questions using only the content of an Agentic AI ebook. The PDF is split into chunks, embedded with Google Gemini, stored in Pinecone, and queried through a LangGraph workflow exposed as a FastAPI service.

If a question can't be answered from the book, the agent says so instead of making something up.

How it works
                 ┌──────────────┐
  PDF ebook ───► │  Ingestion   │  load → split into chunks → embed → upsert
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   Pinecone   │  vector index of book chunks
                 └──────┬───────┘
                        ▲
  User question ─► FastAPI /chat ─► LangGraph:  retrieve ──► generate ──► answer
Ingestion (src/ingestion.py): loads the PDF, splits it into overlapping chunks, creates embeddings, and uploads them to a Pinecone index.
Retrieval: the user's question is embedded and the most similar chunks are fetched from Pinecone.
Generation (src/graph.py): a LangGraph pipeline passes the retrieved chunks to a Gemini model, which answers strictly from that context, with guardrails for out-of-scope questions.
API (app.py): a FastAPI app exposes the agent over HTTP.
Tech stack
Language: Python 3.13
Orchestration: LangChain, LangGraph
LLM: Google Gemini (via langchain-google-genai)
Embeddings: gemini-embedding-001 (3072 dimensions)
Vector database: Pinecone
API server: FastAPI + Uvicorn
PDF parsing: PyPDF
Project structure
rag-ai-agent-new/
├── app.py                  # FastAPI application (/chat endpoint)
├── check_models.py         # Helper: lists Gemini models available to your API key
├── tests_sample_queries.py # Sample questions for testing the agent
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .gitignore
├── data/
│   └── Ebook-Agentic-AI.pdf   # Source document
└── src/
    ├── __init__.py
    ├── config.py           # Model names, index name, chunk settings, API keys
    ├── ingestion.py        # PDF → chunks → embeddings → Pinecone
    └── graph.py            # LangGraph retrieve + generate workflow
Getting started
Prerequisites
Python 3.10 or newer
A Google AI Studio API key
A Pinecone API key
1. Clone the repository
bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
2. Create and activate a virtual environment

Windows (PowerShell):

powershell
python -m venv venv
venv\Scripts\Activate.ps1

macOS / Linux:

bash
python -m venv venv
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Configure environment variables

Copy the example file and fill in your keys:

bash
cp .env.example .env

On Windows PowerShell use Copy-Item .env.example .env.

env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key

Never commit your real .env file. It is listed in .gitignore.

5. Ingest the ebook (run once)
bash
python -m src.ingestion

This creates the Pinecone index (if it doesn't exist), then loads, chunks, embeds, and uploads the book. Expect output similar to:

Loaded 60 pages
Created 119 chunks
6. Run the API
bash
uvicorn app:app --reload

Open http://127.0.0.1:8000/docs to try the /chat endpoint from the interactive Swagger UI.

Note: --reload only watches .py files. If you change .env, stop the server (Ctrl+C) and start it again.

Example usage
bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is an AI agent?"}'

The request field name may differ; check the /docs page for the exact schema used by app.py.

You can also run the prepared sample questions:

bash
python tests_sample_queries.py
Configuration

Key settings live in src/config.py:

Setting	Purpose
EMBED_MODEL	Embedding model (models/gemini-embedding-001)
EMBED_DIM	Embedding size, which must match the Pinecone index dimension (3072)
LLM_MODEL	Gemini model used to write answers
INDEX_NAME	Pinecone index name (agentic-ai-index)

If you change the embedding model, its output dimension may change too. Delete the old Pinecone index and re-run ingestion so the dimensions match.

Troubleshooting
404 NOT_FOUND ... is not found for API version v1beta: the model name isn't available to your API key. Run python check_models.py to list the models your key supports, then update config.py.
429 / quota errors: the free tier is rate-limited. Wait a minute, or switch to a lighter model such as a flash-lite variant.
Dimension mismatch error from Pinecone: the index was created with a different dimension than your embedding model. Delete the index in the Pinecone console and re-run ingestion.
Changed .env but nothing changed: restart Uvicorn manually.
Deployment

The app can be hosted on a platform such as Render:

Build command: pip install -r requirements.txt
Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
Environment variables: set GOOGLE_API_KEY and PINECONE_API_KEY in the dashboard, using the same Pinecone key and index you used for ingestion.
Limitations
Answers are limited to the content of the ingested PDF.
Quality depends on chunking and retrieval; very broad questions may return partial answers.
Free-tier API limits may throttle heavy use.
