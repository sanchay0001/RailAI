"""
main.py
--------
FastAPI application — the HTTP layer of RailAI. Every request from the
frontend passes through this file. It is intentionally thin: it validates
inputs, delegates all real work to rag_pipeline.py and document_processor.py,
and formats responses consistently.

Run locally with:
    uvicorn main:app --reload --port 8000

The --reload flag means uvicorn auto-restarts when you save any .py file,
which is useful during development.
"""

import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
import document_processor
import vector_store
import rag_pipeline


# ---------------------------------------------------------------------------
# LIFESPAN — replaces the deprecated @app.on_event("startup") pattern.
# ---------------------------------------------------------------------------
# asynccontextmanager turns this function into a context manager that
# FastAPI calls automatically. Code BEFORE "yield" runs at startup;
# code AFTER "yield" (if any) runs at shutdown. This is the modern
# FastAPI pattern for startup/shutdown logic as of FastAPI 0.93+.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs automatically when FastAPI starts (both locally and on Render).
    Scans data/ for any PDFs not yet in ChromaDB and indexes them.

    On Render with a persistent disk: this does nothing after the first
    deploy because all PDFs are already indexed and metadata.json records
    their hashes — so the hash check skips everything and startup is fast.

    On a fresh Render deploy (or after clearing chroma_db): this fully
    re-indexes everything — the intended self-healing behaviour so you
    never have to manually trigger indexing on the server.
    """
    print("🚀 RailAI starting up — checking knowledge base...")
    try:
        new_chunks = document_processor.process_all_documents()
        if new_chunks:
            vector_store.add_chunks_to_store(new_chunks)
            print(f"✅ Startup indexed {len(new_chunks)} new chunks.")
        else:
            total = vector_store.get_total_indexed_chunk_count()
            print(f"✅ Knowledge base already up to date ({total} chunks ready).")
    except Exception as e:
        # Catch but don't crash on startup errors — the server should
        # still start even if indexing fails, so /health can report the
        # problem rather than the whole process dying silently.
        print(f"⚠️  Startup indexing error: {e}")

    yield  # App runs here — everything after yield is shutdown logic


# ---------------------------------------------------------------------------
# APP INITIALISATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RailAI — Railway Employee Help Desk",
    description="Internal AI assistant for Indian Railways employees.",
    version="1.0.0",
    # Pass the lifespan context manager here instead of using on_event.
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS MIDDLEWARE
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SERVE FRONTEND STATIC FILES
# ---------------------------------------------------------------------------
FRONTEND_DIR = config.BASE_DIR / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS (Pydantic)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Body expected by POST /chat"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The employee's question to the AI assistant.",
        # json_schema_extra is the Pydantic v2 replacement for the
        # deprecated "example" keyword argument on Field().
        json_schema_extra={"example": "What should I do if the signal system fails?"},
    )
    department: str | None = Field(
        default=None,
        description="Department key to filter search (e.g. 'signal'). "
                    "If null, all departments are searched.",
        json_schema_extra={"example": "signal"},
    )


class ChatResponse(BaseModel):
    """Shape of the JSON body returned by POST /chat"""
    answer: str
    sources: list
    department: str
    chunks_used: int


class ReindexResponse(BaseModel):
    """Shape of the JSON body returned by POST /reindex"""
    message: str
    new_chunks_added: int
    total_chunks: int


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    """
    Serves the main HTML page. When a user visits the Render URL (or
    localhost:8000), they get the chat interface — not a raw JSON response.
    """
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health_check():
    """
    Simple health endpoint — useful for Render's health check pings and
    for quickly verifying the server is alive and the knowledge base has
    chunks.
    """
    try:
        total_chunks = vector_store.get_total_indexed_chunk_count()
        indexed_docs = document_processor.get_indexed_documents_summary()
        return {
            "status": "ok",
            "total_chunks": total_chunks,
            "total_documents": len(indexed_docs),
            "model": config.GROQ_MODEL_NAME,
            "departments": list(config.DEPARTMENTS.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/departments")
async def get_departments():
    """
    Returns the department list for the frontend dropdown.
    Called once on page load to populate the selector.
    """
    departments = [
        {"key": key, "label": label}
        for key, label in config.DEPARTMENTS.items()
    ]
    return {
        "departments": [{"key": "", "label": "All Departments"}] + departments
    }


@app.get("/documents")
async def list_documents():
    """
    Returns a summary of all indexed documents with chunk counts.
    """
    docs = document_processor.get_indexed_documents_summary()
    total_chunks = vector_store.get_total_indexed_chunk_count()
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": total_chunks,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    The main endpoint — receives a question + optional department and
    returns an AI-generated answer with source references.
    """
    try:
        result = rag_pipeline.answer_question(
            question=request.question,
            department=request.department if request.department else None,
        )

        if "error" in result and result["error"] == "no_chunks_found":
            raise HTTPException(
                status_code=404,
                detail=result["answer"],
            )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            department=result["department"],
            chunks_used=result["chunks_used"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while generating answer: {str(e)}",
        )


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    department: str = "operations",
):
    """
    Accepts a PDF upload, saves it to data/ with the correct department
    prefix, then triggers re-indexing so it's immediately searchable.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if department not in config.DEPARTMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{department}'. "
                   f"Valid options: {list(config.DEPARTMENTS.keys())}",
        )

    save_path = config.DATA_DIR / f"{department}_{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    new_chunks = document_processor.process_all_documents()
    if new_chunks:
        vector_store.add_chunks_to_store(new_chunks)

    return {
        "message": f"File '{file.filename}' uploaded and indexed successfully.",
        "saved_as": save_path.name,
        "new_chunks": len(new_chunks),
    }


@app.post("/reindex", response_model=ReindexResponse)
async def reindex():
    """
    Manually triggers a full re-scan of data/ and indexes any new or
    changed PDFs.
    """
    try:
        new_chunks = document_processor.process_all_documents()
        added = 0
        if new_chunks:
            added = vector_store.add_chunks_to_store(new_chunks)
        total = vector_store.get_total_indexed_chunk_count()
        return ReindexResponse(
            message=(
                f"Re-indexing complete. {added} new chunks added."
                if added else
                "All documents already up to date. No new chunks added."
            ),
            new_chunks_added=added,
            total_chunks=total,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))