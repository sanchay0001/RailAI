"""
vector_store.py
----------------
Wraps all interaction with ChromaDB. Uses FastEmbeddings instead of
HuggingFace local embeddings — FastEmbed uses a much lighter model
(~50MB vs 500MB) making it suitable for free-tier cloud deployment.
"""

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

import config

# ---------------------------------------------------------------------------
# LAZY EMBEDDING MODEL
# ---------------------------------------------------------------------------
_embedding_model = None


def get_embedding_model():
    """
    Returns the FastEmbed embedding model, initialising it on first call.
    FastEmbed downloads a ~50MB model (vs HuggingFace's ~500MB),
    making it viable on Render/Railway free tiers.
    """
    global _embedding_model
    if _embedding_model is None:
        print("📥 Loading FastEmbed embedding model...")
        # BAAI/bge-small-en-v1.5 is a high quality small embedding model.
        # It produces 384-dimensional vectors, same as all-MiniLM-L6-v2,
        # so ChromaDB storage is identical in structure.
        _embedding_model = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        print("✅ FastEmbed model loaded.")
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
    """
    store = get_vector_store()
    return store._collection.count()