import sys
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What are the challenges or risks of deploying Agentic AI?",
    "Who won the 2022 FIFA World Cup?",
]

for q in QUERIES:
    r = requests.post(f"{BASE_URL}/chat", json={"query": q}, timeout=180)
    data = r.json()
    print("=" * 70)
    print("Q:", q)
    print("A:", data["answer"])
    print("Confidence:", data["confidence_score"])
    print("Chunks retrieved:", len(data["retrieved_chunks"]))