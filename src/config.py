import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-flash-latest"
EMBED_DIM = 3072

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

MIN_RELEVANCE = 0.25