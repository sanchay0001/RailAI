"""
run_indexing.py
----------------
Standalone script — run this directly to process all PDFs in data/ and
store their embeddings in ChromaDB. This exists so we can test Phase 2
(document processing + vector storage) WITHOUT needing the FastAPI server
or frontend to exist yet.

Usage (from the backend/ folder):
    python run_indexing.py
"""

import document_processor
import vector_store


def main():
    print("=" * 60)
    print("RailAI — Knowledge Base Indexing")
    print("=" * 60)

    # Step 1: scan data/, extract text, chunk it, tag departments.
    # This returns ONLY new/changed chunks (already-indexed files are skipped).
    new_chunks = document_processor.process_all_documents()

    # Step 2: embed those chunks and store them in ChromaDB.
    vector_store.add_chunks_to_store(new_chunks)

    # Step 3: print a final summary so you can visually confirm it worked.
    total_chunks = vector_store.get_total_indexed_chunk_count()
    print("-" * 60)
    print(f"📊 Total chunks now in ChromaDB: {total_chunks}")

    print("\n📁 Indexed documents summary:")
    for doc in document_processor.get_indexed_documents_summary():
        print(
            f"   • {doc['filename']:45s} "
            f"[{doc['department']:12s}] "
            f"{doc['num_pages']} pages, {doc['num_chunks']} chunks"
        )

    print("=" * 60)
    print("✅ Indexing complete.")


if __name__ == "__main__":
    main()