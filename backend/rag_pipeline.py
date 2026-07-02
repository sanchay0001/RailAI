"""
rag_pipeline.py
----------------
The core intelligence layer of RailAI. This module connects the two halves
of the system together:
  LEFT  HALF: ChromaDB vector store (already built in Phase 2)
  RIGHT HALF: Groq LLM that generates the final answer

The RAG (Retrieval Augmented Generation) flow:
  1. Receive user question + optional department filter
  2. Search ChromaDB for the most semantically similar chunks
  3. Format those chunks into a readable context block
  4. Build a structured prompt (system instructions + context + question)
  5. Send prompt to Groq LLM and stream back the response
  6. Return a structured dict: answer text + list of source references

Nothing in this file touches the filesystem directly — it calls
vector_store.py for retrieval and uses the Groq API for generation.
"""

from langchain_groq import ChatGroq
# In LangChain 1.x, message classes moved from langchain.schema
# to langchain_core.messages — this is the correct import path.
from langchain_core.messages import HumanMessage, SystemMessage

import config
import vector_store


def _format_context_from_chunks(chunks: list) -> tuple:
    """
    Converts a list of retrieved LangChain Document chunks into:
      1. A single formatted context string to inject into the LLM prompt
      2. A clean list of source references to return to the frontend

    Each chunk in the context is labeled with its source file and page
    number so the LLM can naturally reference them in its answer, and
    so we can show the user "Source: signal_manual.pdf, page 14" in the UI.

    Args:
        chunks: list of LangChain Document objects from similarity_search()

    Returns:
        (context_string, sources_list)
        - context_string: formatted text block for the LLM prompt
        - sources_list: list of dicts with source/page/department for the UI
    """
    context_parts = []
    sources = []
    # Track which (source, page) combos we have already added so we do not
    # list the same page twice in the sources panel (duplicates happen
    # when multiple chunks from the same page are top-K matches).
    seen_sources = set()

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "Unknown document")
        page = chunk.metadata.get("page", "?")
        department = chunk.metadata.get("department", "general")

        # Format each chunk as a numbered excerpt with clear attribution.
        # The LLM will use these labels when it writes things like
        # "According to Excerpt 2 (signal_manual.pdf, page 45)..."
        context_parts.append(
            f"[Excerpt {i} | Source: {source} | Page: {page}]\n"
            f"{chunk.page_content.strip()}"
        )

        # Build the deduplicated sources list for the frontend sidebar.
        source_key = (source, page)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "filename": source,
                "page": page,
                "department": department,
            })

    # Join all excerpts with a clear visual separator so the LLM can
    # easily distinguish where one chunk ends and another begins.
    context_string = "\n\n---\n\n".join(context_parts)
    return context_string, sources


def _build_prompt_messages(question: str, context: str) -> list:
    """
    Builds the list of messages sent to the Groq chat model.

    We use a two-message structure:
      - SystemMessage: establishes RailAI's identity and strict rules
        (stay grounded in context, admit uncertainty, be precise for
        safety-critical procedures). This is the SYSTEM_PROMPT from config.
      - HumanMessage: the actual question, with the retrieved context
        pasted in above it so the model can reference specific excerpts.

    The context is placed BEFORE the question inside the HumanMessage
    (not in the system prompt) because that is where most LLMs attend to
    it most reliably — system prompt is for persona/rules, user turn
    is for the actual task content.

    Args:
        question: the employee's question string
        context: formatted context block from _format_context_from_chunks()

    Returns:
        list of LangChain message objects ready to send to ChatGroq
    """
    return [
        # config.SYSTEM_PROMPT defines RailAI's persona, rules, and tone.
        SystemMessage(content=config.SYSTEM_PROMPT),

        # The human turn contains the retrieved context plus the question.
        # We explicitly label the sections so the model understands the
        # structure: here is your evidence, here is the question to answer.
        HumanMessage(content=(
            f"RELEVANT DOCUMENT EXCERPTS:\n"
            f"{'=' * 50}\n"
            f"{context}\n"
            f"{'=' * 50}\n\n"
            f"EMPLOYEE QUESTION:\n{question}"
        )),
    ]


def get_llm() -> ChatGroq:
    """
    Initializes and returns a ChatGroq instance.

    Called once per request (not at module load time) because we want
    startup failures (e.g. missing API key) to surface as clear API
    errors, not as silent import-time crashes that are hard to debug.
    """
    if not config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file locally, "
            "or to Render's environment variables in the dashboard."
        )

    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        # llama-3.1-8b-instant: fast, cheap, excellent at following
        # structured instructions. Well-suited for a help-desk tool
        # where speed and factual accuracy matter more than creativity.
        model=config.GROQ_MODEL_NAME,
        # Low temperature = deterministic, factual, consistent answers.
        # We never want the model to creatively invent a railway procedure.
        temperature=config.LLM_TEMPERATURE,
        # Cap output length. Railway SOP answers should be concise,
        # not multi-page essays. 1024 tokens is approximately 750 words.
        max_tokens=config.LLM_MAX_TOKENS,
    )


def answer_question(question: str, department: str = None) -> dict:
    """
    Main entry point for the RAG pipeline. Takes a user question and an
    optional department filter, returns a fully structured answer dict.

    Args:
        question:   the employee's natural-language question
        department: optional string key from config.DEPARTMENTS (e.g. "signal").
                    If None, the search covers all departments.

    Returns a dict with these keys:
        answer      (str)  - the LLM-generated answer text
        sources     (list) - list of {filename, page, department} dicts
        department  (str)  - the department that was searched (or "all")
        chunks_used (int)  - how many document chunks were fed to the LLM
        error       (str)  - only present if something went wrong
    """
    # Validate department key if one was provided — reject anything that
    # is not a known department key to prevent garbage metadata filters
    # from silently returning zero results.
    if department and department not in config.DEPARTMENTS:
        return {
            "answer": (
                f"Unknown department '{department}'. "
                f"Valid options: {list(config.DEPARTMENTS.keys())}"
            ),
            "sources": [],
            "department": department,
            "chunks_used": 0,
            "error": "invalid_department",
        }

    # ----------------------------------------------------------------
    # STEP 1: Retrieve the most relevant chunks from ChromaDB
    # ----------------------------------------------------------------
    # similarity_search handles the department filter internally — if
    # department is None it searches everything; otherwise it restricts
    # the search to chunks tagged with that department in their metadata.
    chunks = vector_store.similarity_search(
        query=question,
        department=department,
    )

    # If ChromaDB returned nothing (e.g. knowledge base is empty, or the
    # department has no indexed documents), return a helpful message instead
    # of sending an empty context to the LLM (which would cause hallucinations).
    if not chunks:
        dept_label = config.DEPARTMENTS.get(department, "all departments")
        return {
            "answer": (
                f"No relevant documents found in the knowledge base"
                f"{f' for the {dept_label} department' if department else ''}. "
                f"Please ensure PDFs have been indexed, or try a different department."
            ),
            "sources": [],
            "department": department or "all",
            "chunks_used": 0,
            "error": "no_chunks_found",
        }

    # ----------------------------------------------------------------
    # STEP 2: Format retrieved chunks into context + source references
    # ----------------------------------------------------------------
    context, sources = _format_context_from_chunks(chunks)

    # ----------------------------------------------------------------
    # STEP 3: Build the prompt messages for the LLM
    # ----------------------------------------------------------------
    messages = _build_prompt_messages(question, context)

    # ----------------------------------------------------------------
    # STEP 4: Send to Groq and get the answer
    # ----------------------------------------------------------------
    llm = get_llm()

    # .invoke() sends the messages and returns an AIMessage object.
    # .content extracts the raw text string from that message object.
    response = llm.invoke(messages)
    answer_text = response.content.strip()

    # ----------------------------------------------------------------
    # STEP 5: Return fully structured result
    # ----------------------------------------------------------------
    return {
        "answer": answer_text,
        "sources": sources,
        # Return the human-readable department label, not just the key,
        # so the frontend can display "Searched: Signal department".
        "department": config.DEPARTMENTS.get(department, "All Departments"),
        "chunks_used": len(chunks),
    }