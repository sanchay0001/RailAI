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

import asyncio
import shutil
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
# LIFESPAN — modern FastAPI startup/shutdown pattern (replaces on_event).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI calls everything BEFORE yield at startup, and everything
    AFTER yield at shutdown.

    The key fix for Render: we do NOT run indexing synchronously here.
    If we did, Render's port-binding timeout (60 seconds) would expire
    before uvicorn finishes embedding 7,966 chunks, and the deploy would
    fail with "No open ports detected".

    Instead we fire the indexing off as a background asyncio task. The
    server binds to the port IMMEDIATELY, Render sees it as healthy, and
    indexing runs quietly in the background. Since we're committing the
    pre-built chroma_db/ to the repo, the background task will find
    everything already indexed and finish in under 2 seconds anyway.
    """

    async def index_in_background():
        # Small delay ensures uvicorn has fully bound the port before we
        # start consuming CPU — avoids any race condition on slow machines.
        await asyncio.sleep(2)
        print("🚀 RailAI — checking knowledge base in background...")
        try:
            # run_in_executor runs synchronous (blocking) functions in a
            # thread pool so they don't block the async event loop. Without
            # this, a long indexing run would freeze the server and make it
            # unable to respond to any HTTP requests during that time.
            loop = asyncio.get_event_loop()

            new_chunks = await loop.run_in_executor(
                None,  # None = use the default ThreadPoolExecutor
                document_processor.process_all_documents
            )

            if new_chunks:
                # add_chunks_to_store takes a positional argument, so we
                # wrap it in a lambda to pass it cleanly to run_in_executor.
                await loop.run_in_executor(
                    None,
                    lambda: vector_store.add_chunks_to_store(new_chunks)
                )
                print(f"✅ Background indexing complete: {len(new_chunks)} new chunks added.")
            else:
                total = vector_store.get_total_indexed_chunk_count()
                print(f"✅ Knowledge base already up to date ({total} chunks ready).")

        except Exception as e:
            # Log but never crash — a broken index should not kill the
            # server process. The /health endpoint will report 0 chunks
            # which makes the problem visible without a total outage.
            print(f"⚠️  Background indexing error: {e}")

    # asyncio.create_task() schedules index_in_background() to run
    # concurrently without blocking this function from returning.
    # The server starts accepting requests immediately after this line.
    asyncio.create_task(index_in_background())

    yield  # Server is live and handling requests here


# ---------------------------------------------------------------------------
# APP INITIALISATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RailAI — Railway Employee Help Desk",
    description="Internal AI assistant for Indian Railways employees.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS MIDDLEWARE
# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) controls which origins (domains) are
# allowed to call this API from a browser. allow_origins=["*"] permits
# requests from any domain — safe here since this is an internal tool
# and all data comes from authenticated Groq API calls anyway.
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
# Mounts the frontend/ folder so FastAPI serves HTML/CSS/JS directly.
# This means only ONE process is needed on Render — no separate web server.
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
    Serves the main HTML page. Visiting the Render URL returns the chat
    UI, not a raw JSON response.
    """
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health_check():
    """
    Health endpoint — used by Render's health check pings and for
    verifying the server is alive with chunks indexed.
    Returns 200 when healthy, 503 if the vector store is broken.
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
    Called once on page load — never hardcoded in the HTML.
    """
    departments = [
        {"key": key, "label": label}
        for key, label in config.DEPARTMENTS.items()
    ]
    # Prepend "All Departments" so employees can search globally.
    return {
        "departments": [{"key": "", "label": "All Departments"}] + departments
    }


@app.get("/documents")
async def list_documents():
    """
    Returns a summary of all indexed documents with page and chunk counts.
    Used by the frontend sidebar to show what's in the knowledge base.
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
    Main endpoint — receives a question + optional department filter,
    returns an AI-generated answer with source references.

    FastAPI automatically validates the request body against ChatRequest:
      - Rejects questions shorter than 3 chars with 422
      - Rejects missing question field with 422
    """
    try:
        result = rag_pipeline.answer_question(
            question=request.question,
            # Pass None if department is empty string (All Departments)
            # so similarity_search() skips the metadata filter entirely.
            department=request.department if request.department else None,
        )

        # Surface "no documents found" as a 404 so the frontend can
        # display a clear message instead of showing an empty answer.
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
        # Re-raise HTTP exceptions unchanged — already formatted correctly.
        raise
    except Exception as e:
        # Catch unexpected errors (Groq API down, ChromaDB issue, etc.)
        # and return a clean 500 instead of an ugly Python traceback.
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

    # Save with department prefix so process_all_documents() auto-detects
    # the department from the filename on next indexing run.
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
    changed PDFs. Already-indexed unchanged files are skipped via hash
    check, so this is safe to call at any time without duplicating data.
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