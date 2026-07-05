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

# BASE_DIR is computed FIRST — before load_dotenv() — because we use it
# to find the .env file by absolute path. This ensures load_dotenv() always
# finds the .env file in the project root regardless of which directory
# Python is run from (e.g. running from backend/ vs project root).
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the project root using its absolute path.
# This replaces the bare load_dotenv() call which only searched the
# current working directory — causing GOOGLE_API_KEY to not load when
# run_indexing.py was run from the backend/ subdirectory.
load_dotenv(dotenv_path=BASE_DIR / ".env")

# -----------------------------------------------------------------------
# BASE PATHS
# -----------------------------------------------------------------------
DATA_DIR     = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
METADATA_FILE = DATA_DIR / "metadata.json"

# -----------------------------------------------------------------------
# GROQ LLM CONFIGURATION
# -----------------------------------------------------------------------
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 1024

# -----------------------------------------------------------------------
# GOOGLE EMBEDDING CONFIGURATION
# -----------------------------------------------------------------------
# Google's text-embedding-004 runs entirely via API — zero local RAM usage.
# Free tier: 1500 requests/minute. Get a key at https://aistudio.google.com/apikey
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2", "")
GOOGLE_API_KEY_3 = os.getenv("GOOGLE_API_KEY_3", "")
# Ollama runs as a local server. Override with env var on Render if needed.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# -----------------------------------------------------------------------
# TEXT CHUNKING CONFIGURATION
# -----------------------------------------------------------------------
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# -----------------------------------------------------------------------
# RETRIEVAL CONFIGURATION
# -----------------------------------------------------------------------
TOP_K_RESULTS = 4

# -----------------------------------------------------------------------
# DEPARTMENTS
# -----------------------------------------------------------------------
DEPARTMENTS = {
    "signal":       "Signal",
    "electrical":   "Electrical",
    "it":           "IT / Server",
    "operations":   "Operations",
    "announcement": "Announcement System",
    "safety":       "Safety",
}

# -----------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------
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