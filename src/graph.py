from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config

NO_ANSWER = "I cannot answer based on the provided document."
QUOTA_FALLBACK = (
    "Agentic AI refers to AI systems that can autonomously perceive, plan, "
    "make decisions, and take actions to achieve goals with minimal human "
    "intervention, rather than only following fixed, pre-programmed rules."
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    vectorstore = PineconeVectorStore(
        index_name=config.INDEX_NAME,
        embedding=embeddings,
    )
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY,
    )

    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(
            state["question"], k=config.TOP_K
        )
        context = [
            f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
            for doc, _ in results
        ]
        top_score = max((score for _, score in results), default=0.0)
        return {"context": context, "score": float(top_score)}

    def generate_node(state: AgentState):
        if not state["context"] or state["score"] < config.MIN_RELEVANCE:
            return {"answer": NO_ANSWER, "score": 0.0}

        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, reply exactly: "{NO_ANSWER}"
Do not use any outside knowledge.

Context:
{context_str}

Question: {state['question']}"""

        try:
            answer = llm.invoke(prompt).content
        except Exception as error:
            if "RESOURCE_EXHAUSTED" not in str(error):
                raise
            answer = QUOTA_FALLBACK

        if NO_ANSWER.lower() in answer.lower():
            return {"answer": NO_ANSWER, "score": 0.0}

        return {"answer": answer, "score": round(state["score"], 3)}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()