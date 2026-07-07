"""
vector_store.py
----------------
Uses Google's gemini-embedding-001 via API with round-robin across
multiple keys to stay under rate limits. This version is used for
BOTH local indexing and Render production — ensuring the same embedding
space is used for stored vectors and query vectors.
"""

import time
import os
from itertools import cycle

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import config


def get_api_keys() -> list:
    """Reads all available Google API keys from environment."""
    keys = []
    if os.getenv("GOOGLE_API_KEY"):
        keys.append(os.getenv("GOOGLE_API_KEY"))
    for i in range(2, 10):
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            keys.append(key)
    if not keys:
        raise ValueError("No Google API keys found. Set GOOGLE_API_KEY in your .env file.")
    print(f"🔑 Found {len(keys)} Google API key(s) for round-robin embedding.")
    return keys


def make_embedding_model(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Creates an embedding model instance with the given API key."""
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )


_embedding_model = None


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Returns the primary embedding model for search queries."""
    global _embedding_model
    if _embedding_model is None:
        keys = get_api_keys()
        # On Render, GOOGLE_API_KEY is set in dashboard environment variables.
        # Locally, it's read from .env file.
        _embedding_model = make_embedding_model(keys[0])
    return _embedding_model


def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=str(config.CHROMA_DB_DIR),
        embedding_function=get_embedding_model(),
        collection_name="railai_documents",
    )


def add_chunks_to_store(chunks: list, start_from: int = 0) -> int:
    """
    Embeds and stores chunks using round-robin across all API keys.
    25s pause between batches keeps each key under 100/minute limit.
    """
    if not chunks:
        print("ℹ️  No new chunks to add to vector store.")
        return 0

    keys = get_api_keys()
    key_cycle = cycle(keys)

    BATCH_SIZE = 80
    total_added = 0
    batch_num = 0

    chunks_to_process = chunks[start_from:]
    if start_from > 0:
        print(f"▶️  Resuming from chunk {start_from}, {len(chunks_to_process)} remaining...")

    for i in range(0, len(chunks_to_process), BATCH_SIZE):
        batch = chunks_to_process[i: i + BATCH_SIZE]
        batch_num += 1

        current_key = next(key_cycle)
        key_index = keys.index(current_key) + 1
        print(f"   Batch {batch_num}: using key #{key_index}")

        store = Chroma(
            persist_directory=str(config.CHROMA_DB_DIR),
            embedding_function=make_embedding_model(current_key),
            collection_name="railai_documents",
        )

        store.add_documents(batch)
        total_added += len(batch)
        print(f"   …embedded and stored {start_from + total_added}/{start_from + len(chunks_to_process)} chunks")

        if total_added < len(chunks_to_process):
            print("   ⏳ Pausing 25s...")
            time.sleep(25)

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