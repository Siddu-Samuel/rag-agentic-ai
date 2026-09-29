import time

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config

BATCH_SIZE = 50       # chunks per batch, safely under the 100/min limit
PAUSE_SECONDS = 60    # wait this long between batches


def ensure_index():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]

    if config.INDEX_NAME not in existing:
        print(f"Creating index '{config.INDEX_NAME}'...")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(config.INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        print(f"Index '{config.INDEX_NAME}' already exists.")


def run_ingestion():
    ensure_index()

    loader = PyPDFLoader(config.PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )

    vectorstore = PineconeVectorStore(
        index_name=config.INDEX_NAME,
        embedding=embeddings,
    )

    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

        if i + BATCH_SIZE < total:
            print(f"Pausing {PAUSE_SECONDS}s to respect the free-tier rate limit...")
            time.sleep(PAUSE_SECONDS)

    print("Upserted all chunks to Pinecone")


if __name__ == "__main__":
    run_ingestion()