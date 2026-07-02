"""
vector_store.py
----------------
Wraps all interaction with ChromaDB so the rest of the app never has to
deal with ChromaDB's API directly. Responsibilities:
  1. Initialize (or load) the persistent ChromaDB collection on disk.
  2. Embed and store new document chunks.
  3. Run similarity search for a user query, optionally filtered by department.
"""

# HuggingFaceEmbeddings wraps a local sentence-transformers model so
# LangChain can call it the same way it would call any embedding provider.
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma is LangChain's wrapper around the chromadb library — it handles
# storing Document objects (text + metadata) as vectors automatically.
# We import from the dedicated `langchain_chroma` package (not
# langchain_community), since the community version is deprecated.
from langchain_chroma import Chroma

import config


# -----------------------------------------------------------------------
# EMBEDDING MODEL — loaded once at module import time, not per-request.
# -----------------------------------------------------------------------
# Loading a sentence-transformers model takes a second or two, so we do it
# ONCE when this module is first imported (at server startup), not on every
# single chat request — that would make every query painfully slow.
_embedding_model = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME,
    # model_kwargs forces CPU usage explicitly — Render's free/standard
    # tiers don't have GPUs, and this avoids any auto-detection overhead.
    model_kwargs={"device": "cpu"},
)


def get_vector_store() -> Chroma:
    """
    Returns a Chroma vector store instance connected to our persistent
    on-disk database. Calling this multiple times reconnects to the SAME
    underlying data — it does not create a new empty store each time,
    as long as persist_directory points to the same folder.
    """
    return Chroma(
        # Where ChromaDB writes its index files. On Render this path will
        # live on the persistent disk, so data survives restarts.
        persist_directory=str(config.CHROMA_DB_DIR),
        # The embedding function ChromaDB uses both when STORING new
        # chunks and when EMBEDDING a user's query for similarity search —
        # using the same model for both is essential for accurate matching.
        embedding_function=_embedding_model,
        # A named collection (like a table) inside the ChromaDB store.
        collection_name="railai_documents",
    )


def add_chunks_to_store(chunks: list) -> int:
    """
    Embeds and stores a list of LangChain Document chunks into ChromaDB.
    Called by the indexing pipeline after document_processor.py produces
    new chunks. Returns the number of chunks added.

    If chunks is empty (e.g. nothing new to index), this does nothing —
    avoids an unnecessary empty call to Chroma.

    IMPORTANT: ChromaDB enforces a maximum batch size per upsert() call
    (commonly 5461, derived from SQLite's internal variable limits). Large
    PDFs can easily produce 7000+ chunks in one indexing run, so we split
    the chunks into smaller batches and insert them one batch at a time
    instead of sending everything in a single call.
    """
    if not chunks:
        print("ℹ️  No new chunks to add to vector store.")
        return 0

    store = get_vector_store()

    # Safe batch size — comfortably under ChromaDB's internal limit
    # (~5461), with headroom in case that limit varies slightly by version.
    BATCH_SIZE = 4000

    total_added = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        # add_documents() automatically: embeds each chunk's text using our
        # HuggingFace model, then writes the resulting vector + the chunk's
        # metadata (department, source, page) into the persistent ChromaDB files.
        store.add_documents(batch)
        total_added += len(batch)
        print(f"   …embedded and stored {total_added}/{len(chunks)} chunks")

    print(f"✅ Added {total_added} chunks to ChromaDB vector store.")
    return total_added


def similarity_search(query: str, department: str = None, k: int = None) -> list:
    """
    Runs a similarity search against ChromaDB for the given user query.

    Args:
        query: the user's natural-language question.
        department: optional department key (e.g. "signal"). If provided,
            ONLY chunks tagged with this department are searched — this is
            how the department dropdown filter actually works under the hood.
        k: how many top results to return. Defaults to config.TOP_K_RESULTS.

    Returns:
        A list of LangChain Document objects (the most relevant chunks),
        each still carrying its .metadata (source, department, page).
    """
    store = get_vector_store()
    k = k or config.TOP_K_RESULTS

    # ChromaDB's metadata filter syntax: {"field": "value"} restricts the
    # search to only vectors whose metadata matches exactly. If department
    # is None (user selected "All Departments"), we pass no filter at all,
    # which searches the entire knowledge base.
    search_filter = {"department": department} if department else None

    # similarity_search() embeds the query internally (using the same
    # embedding model passed in get_vector_store), then returns the k
    # closest-matching chunks by cosine similarity.
    results = store.similarity_search(
        query=query,
        k=k,
        filter=search_filter,
    )

    return results


def get_total_indexed_chunk_count() -> int:
    """
    Returns the total number of chunks currently stored in ChromaDB.
    Used for a simple health/status check (e.g. "Knowledge base: 1,204
    chunks indexed") shown on the frontend.
    """
    store = get_vector_store()
    # The underlying chromadb collection object exposes .count() directly.
    return store._collection.count()