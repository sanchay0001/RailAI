"""
vector_store.py
----------------
Uses Ollama's nomic-embed-text model for embeddings.
Runs locally via Ollama — no API keys, no rate limits, no RAM issues.
The quantized model uses only ~275MB RAM.
"""

import time
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config

_embedding_model = None


def get_embedding_model() -> OllamaEmbeddings:
    """
    Returns the Ollama embedding model, initialising it on first call.
    Ollama must be running locally (auto-starts on Windows after install).
    """
    global _embedding_model
    if _embedding_model is None:
        print("📥 Initialising Ollama nomic-embed-text model...")
        _embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=config.OLLAMA_HOST,
        )
        print("✅ Ollama embedding model ready.")
    return _embedding_model


def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=str(config.CHROMA_DB_DIR),
        embedding_function=get_embedding_model(),
        collection_name="railai_documents",
    )


def add_chunks_to_store(chunks: list, start_from: int = 0) -> int:
    """
    Embeds and stores chunks into ChromaDB.
    No rate limits — Ollama runs locally so batches can be large and fast.
    """
    if not chunks:
        print("ℹ️  No new chunks to add to vector store.")
        return 0

    store = get_vector_store()
    BATCH_SIZE = 200
    total_added = 0

    chunks_to_process = chunks[start_from:]
    if start_from > 0:
        print(f"▶️  Resuming from chunk {start_from}, {len(chunks_to_process)} remaining...")

    for i in range(0, len(chunks_to_process), BATCH_SIZE):
        batch = chunks_to_process[i: i + BATCH_SIZE]
        store.add_documents(batch)
        total_added += len(batch)
        print(f"   …embedded and stored {start_from + total_added}/{start_from + len(chunks_to_process)} chunks")

    print(f"✅ Added {total_added} chunks to ChromaDB vector store.")
    return total_added


def similarity_search(query: str, department: str = None, k: int = None) -> list:
    store = get_vector_store()
    k = k or config.TOP_K_RESULTS
    search_filter = {"department": department} if department else None
    return store.similarity_search(query=query, k=k, filter=search_filter)


def get_total_indexed_chunk_count() -> int:
    store = get_vector_store()
    return store._collection.count()