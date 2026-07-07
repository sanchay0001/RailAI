"""
resume_indexing.py
-------------------
Resumes indexing from a specific chunk offset without deleting
already-stored embeddings. Use when indexing was interrupted.

Usage:
    python resume_indexing.py <start_chunk_number>
Example:
    python resume_indexing.py 2640
"""

import sys
import document_processor
import vector_store


def main():
    start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"=" * 60)
    print(f"RailAI — Resuming Indexing from chunk {start_from}")
    print(f"=" * 60)

    # Re-extract all chunks (fast, no API calls)
    # We need to temporarily clear metadata.json so process_all_documents
    # re-extracts text. The existing ChromaDB embeddings are NOT deleted.
    import json
    import config

    # Read existing metadata, clear it temporarily to force re-extraction
    metadata_backup = {}
    if config.METADATA_FILE.exists():
        with open(config.METADATA_FILE) as f:
            metadata_backup = json.load(f)
    config.METADATA_FILE.unlink(missing_ok=True)

    # Re-extract text chunks (no embeddings yet)
    all_chunks = document_processor.process_all_documents()

    # Add only from start_from onwards
    added = vector_store.add_chunks_to_store(all_chunks, start_from=start_from)

    total = vector_store.get_total_indexed_chunk_count()
    print(f"\n📊 Total chunks now in ChromaDB: {total}")
    print("=" * 60)
    print("✅ Resume complete.")


if __name__ == "__main__":
    main()