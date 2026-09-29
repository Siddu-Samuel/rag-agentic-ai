from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")

graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "endpoint": "POST /chat"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    result = graph.invoke(
        {"question": request.query, "context": [], "answer": "", "score": 0.0}
    )
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )