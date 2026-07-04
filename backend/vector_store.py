"""
vector_store.py
----------------
Wraps all interaction with ChromaDB. The embedding model is loaded
lazily (on first use) rather than at import time. This prevents the
HuggingFace model download from blocking server startup on Render.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config

# ---------------------------------------------------------------------------
# LAZY EMBEDDING MODEL
# ---------------------------------------------------------------------------
# _embedding_model starts as None. get_embedding_model() initialises it
# on first call and caches it for all subsequent calls. This means the
# ~2 minute model download happens on the first actual request, not at
# import time — so uvicorn starts and binds the port immediately.
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model, initialising it on first call.
    Subsequent calls return the cached instance immediately.
    """
    global _embedding_model
    if _embedding_model is None:
        print("📥 Loading embedding model (first time only)...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
        print("✅ Embedding model loaded.")
    return _embedding_model


def get_vector_store() -> Chroma:
    """
    Returns a Chroma vector store instance connected to our persistent
    on-disk database.
    """
    return Chroma(
        persist_directory=str(config.CHROMA_DB_DIR),
        embedding_function=get_embedding_model(),
        collection_name="railai_documents",
    )


def add_chunks_to_store(chunks: list) -> int:
    """
    Embeds and stores document chunks into ChromaDB in batches.
    Returns the number of chunks added.
    """
    if not chunks:
        print("ℹ️  No new chunks to add to vector store.")
        return 0

    store = get_vector_store()

    # ChromaDB enforces a max batch size (~5461). We use 4000 to be safe.
    BATCH_SIZE = 4000
    total_added = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        store.add_documents(batch)
        total_added += len(batch)
        print(f"   …embedded and stored {total_added}/{len(chunks)} chunks")

    print(f"✅ Added {total_added} chunks to ChromaDB vector store.")
    return total_added


def similarity_search(query: str, department: str = None, k: int = None) -> list:
    """
    Runs a similarity search against ChromaDB for the given user query.

    Args:
        query:      the user's natural-language question
        department: optional department key to filter results
        k:          number of top results to return

    Returns:
        List of LangChain Document objects (most relevant chunks).
    """
    store = get_vector_store()
    k = k or config.TOP_K_RESULTS

    search_filter = {"department": department} if department else None

    results = store.similarity_search(
        query=query,
        k=k,
        filter=search_filter,
    )
    return results


def get_total_indexed_chunk_count() -> int:
    """
    Returns the total number of chunks currently stored in ChromaDB.
    Used for health checks and sidebar stats.
    """
    store = get_vector_store()
    return store._collection.count()