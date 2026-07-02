"""
document_processor.py
----------------------
Responsible for everything that happens to a PDF BEFORE it becomes a vector:
  1. Scan the data/ folder for PDF files.
  2. Determine each PDF's department from its filename prefix.
  3. Load and extract raw text from each PDF (page by page).
  4. Split that text into overlapping chunks suitable for embedding.
  5. Track which files have already been processed in metadata.json so we
     don't waste time/CPU re-processing unchanged files on every startup.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# LangChain's PDF loader — reads a PDF and returns a list of "Document"
# objects, one per page, each with .page_content (text) and .metadata.
from langchain_community.document_loaders import PyPDFLoader

# Splits long text into smaller overlapping chunks. "Recursive" means it
# tries to split on paragraph breaks first, then sentences, then words —
# only falling back to a hard character cut if nothing else fits, which
# keeps chunks semantically coherent instead of cutting mid-sentence.
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def get_department_from_filename(filename: str) -> str:
    """
    Determines department by reading the prefix before the first underscore.
    Example: "signal_axle_counter_maintenance.pdf" -> "signal"

    If the prefix doesn't match any known department key, we default to
    "operations" rather than crashing — better to index it under a broad
    bucket than to skip a document entirely.
    """
    prefix = filename.split("_")[0].lower()
    if prefix in config.DEPARTMENTS:
        return prefix
    return "operations"  # safe fallback for unrecognized prefixes


def compute_file_hash(filepath: Path) -> str:
    """
    Computes an MD5 hash of the file's contents.

    Why we need this: if you replace signal_manual.pdf with an updated
    version but keep the same filename, the OLD metadata entry would
    falsely report "already indexed" and skip re-embedding the new content.
    Hashing the actual bytes means we detect content changes, not just
    filename changes.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read in 8KB chunks instead of the whole file at once — keeps
        # memory usage low even for large PDFs like your 41MB ACTM file.
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_metadata() -> dict:
    """
    Loads metadata.json if it exists, otherwise returns an empty dict.
    This file is our lightweight substitute for a database table that
    would normally track "documents already processed".
    """
    if config.METADATA_FILE.exists():
        with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict) -> None:
    """
    Writes the metadata dict back to metadata.json, pretty-printed so it's
    human-readable if you ever want to inspect it manually.
    """
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def extract_chunks_from_pdf(filepath: Path, department: str) -> list:
    """
    Loads a single PDF, extracts text page by page, and splits it into
    overlapping chunks. Returns a list of LangChain Document objects,
    each chunk carrying metadata: source filename, department, page number.
    """
    # PyPDFLoader reads the PDF and returns one Document per page.
    loader = PyPDFLoader(str(filepath))
    pages = loader.load()

    # The splitter cuts each page's text into ~CHUNK_SIZE character pieces,
    # with CHUNK_OVERLAP characters shared between consecutive chunks so
    # context isn't lost at chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Try splitting on these separators in order of preference —
        # paragraph breaks first, then lines, then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Attach department + clean source filename to every chunk's metadata.
    # This metadata is what lets us later filter ChromaDB searches by
    # department, and show "source: signal_manual.pdf, page 12" in answers.
    for chunk in chunks:
        chunk.metadata["department"] = department
        chunk.metadata["source"] = filepath.name
        # PyPDFLoader already sets "page" in metadata (0-indexed), we just
        # convert it to 1-indexed for human-friendly display later.
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

    return chunks


def process_all_documents() -> list:
    """
    Main entry point for this module. Scans data/ for PDFs, skips files
    that are unchanged since last run (via hash comparison), processes
    new/changed files into chunks, and updates metadata.json.

    Returns a list of ALL new chunks that need to be embedded and stored
    in ChromaDB (empty list if nothing changed since last run).
    """
    metadata = load_metadata()
    all_new_chunks = []

    # Find every PDF file directly inside data/ (not subfolders).
    pdf_files = sorted(config.DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in {config.DATA_DIR}. Add PDFs and re-run.")
        return all_new_chunks

    for filepath in pdf_files:
        filename = filepath.name
        file_hash = compute_file_hash(filepath)

        # Skip this file if we've already processed THIS EXACT content
        # before (same hash recorded in metadata.json).
        existing_entry = metadata.get(filename)
        if existing_entry and existing_entry.get("hash") == file_hash:
            print(f"⏭️  Skipping unchanged file: {filename}")
            continue

        department = get_department_from_filename(filename)
        print(f"📄 Processing: {filename}  →  department: {department}")

        chunks = extract_chunks_from_pdf(filepath, department)
        all_new_chunks.extend(chunks)

        # Record this file as processed, with enough info to skip it next
        # time and to show useful stats in the frontend's document list.
        metadata[filename] = {
            "department": department,
            "hash": file_hash,
            "num_pages": len(set(c.metadata["page"] for c in chunks)),
            "num_chunks": len(chunks),
            "indexed_at": datetime.utcnow().isoformat(),
        }

    save_metadata(metadata)
    print(f"✅ Processed {len(all_new_chunks)} new chunks from {len(pdf_files)} PDF(s) scanned.")
    return all_new_chunks


def get_indexed_documents_summary() -> list:
    """
    Returns a simple list of all indexed documents with their stats, used
    by the GET /documents API endpoint so the frontend can show what's
    been indexed without touching ChromaDB directly.
    """
    metadata = load_metadata()
    summary = []
    for filename, info in metadata.items():
        summary.append({
            "filename": filename,
            "department": info["department"],
            "num_pages": info["num_pages"],
            "num_chunks": info["num_chunks"],
            "indexed_at": info["indexed_at"],
        })
    return summary