"""
config.py
----------
Single source of truth for every constant used across the RailAI backend.
Nothing else in the codebase should hardcode a path, model name, or department
list — everything imports from here. This makes the system easy to tune later
(e.g. changing chunk size) without hunting through multiple files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() reads a local .env file (if present) and injects its key=value
# pairs into os.environ. On Render, environment variables are set directly in
# the dashboard, so this call is harmless there (no .env file exists, it just
# does nothing and Render's real env vars are used instead).
load_dotenv()

# -----------------------------------------------------------------------
# BASE PATHS
# -----------------------------------------------------------------------
# Path(__file__) = path to this config.py file.
# .resolve() = converts it to an absolute path (no "../" ambiguity).
# .parent = the "backend/" folder.
# .parent again = the project root "RailAI/".
# We compute this dynamically instead of hardcoding "C:/Users/..." so the
# project works identically on your Windows machine AND on Render's Linux servers.
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder where all railway PDFs live. document_processor.py scans this folder.
DATA_DIR = BASE_DIR / "data"

# Folder where ChromaDB stores its vector index files on disk.
# On Render, this exact path will be mounted as a persistent disk so the
# embeddings survive service restarts.
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# JSON file that tracks which PDFs have already been indexed, their
# department, page count, and indexing timestamp. This replaces a database.
METADATA_FILE = DATA_DIR / "metadata.json"

# -----------------------------------------------------------------------
# GROQ LLM CONFIGURATION
# -----------------------------------------------------------------------
# The API key is NEVER hardcoded. It's read from the environment, which is
# populated either by your local .env file or by Render's dashboard secrets.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq model used to generate the final answer. llama-3.1-8b-instant is fast
# and cheap — good for a help-desk style Q&A tool where latency matters more
# than maximum reasoning depth.
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Controls randomness of the LLM's output. 0.2 = mostly deterministic and
# factual, which is what we want for an internal SOP assistant (we don't
# want creative or inconsistent answers to safety-critical questions).
LLM_TEMPERATURE = 0.2

# Maximum number of tokens the LLM is allowed to generate in one answer.
LLM_MAX_TOKENS = 1024

# -----------------------------------------------------------------------
# EMBEDDING MODEL CONFIGURATION
# -----------------------------------------------------------------------
# This is a free, local, open-source embedding model that runs on CPU.
# It converts text chunks into vectors (lists of numbers) that capture
# meaning, so ChromaDB can do similarity search. No API key or cost needed.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------------------------------------------------
# TEXT CHUNKING CONFIGURATION
# -----------------------------------------------------------------------
# PDFs are split into smaller overlapping chunks before embedding, because
# embedding an entire 200-page manual as one vector would lose all detail.
# CHUNK_SIZE = max characters per chunk.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP = characters repeated between consecutive chunks, so a
# sentence that gets cut at a chunk boundary still has context on both sides.
CHUNK_OVERLAP = 150

# -----------------------------------------------------------------------
# RETRIEVAL CONFIGURATION
# -----------------------------------------------------------------------
# How many top-matching chunks to retrieve from ChromaDB for each user query.
# 4 is a good balance: enough context for a complete answer, not so much that
# the LLM prompt becomes bloated or noisy with irrelevant chunks.
TOP_K_RESULTS = 4

# -----------------------------------------------------------------------
# DEPARTMENTS
# -----------------------------------------------------------------------
# The five departments employees can filter by. The KEY (e.g. "signal") is
# what gets stored in ChromaDB metadata and matched against filename prefixes.
# The VALUE is the human-readable label shown in the frontend dropdown.
DEPARTMENTS = {
    "signal": "Signal",
    "electrical": "Electrical",
    "it": "IT / Server",
    "operations": "Operations",
    "announcement": "Announcement System",
    "safety": "Safety",
}

# -----------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------
# This instructs the LLM how to behave: stay grounded in retrieved context,
# admit when it doesn't know, and keep the tone appropriate for railway staff.
SYSTEM_PROMPT = """You are RailAI, an internal AI assistant for Indian Railways employees.
You answer technical and operational questions using ONLY the provided document excerpts.

Rules:
1. Base your answer strictly on the provided context. Do not invent procedures.
2. If the context does not contain enough information to answer confidently,
   say so clearly and suggest the employee consult their department manual or supervisor.
3. For safety-critical or emergency procedures, be precise and list steps in order.
4. Keep answers clear and practical — the reader is a working railway employee, not a researcher.
5. Always mention which document(s) your answer is based on, if available in the context.
"""