import os
import shutil
import uuid
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from loaders import load_pdf
from embeddings import hybrid_search
from langgraph_main import build_crag_graph

app = FastAPI(title="CRAG Legal Agent API")

# FastAPI's built-in state container holds the compiled graph in RAM across requests
app.state.crag_app = None


# Pydantic v2 schemas for strict request/response validation
class QueryRequest(BaseModel):
    user_query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    thread_id: str | None = Field(default=None, examples=[None])


class QueryResponse(BaseModel):
    answer: str
    context_used: str
    thread_id: str


class IndexResponse(BaseModel):
    message: str
    chunks_processed: int


@app.get("/health")
async def health_check():
    """
    Pure lightweight check. Uses `async def` so it responds immediately
    without touching the worker threadpool.
    """
    return {
        "status": "healthy",
        "graph_ready": app.state.crag_app is not None,
    }


@app.post("/upload_pdf", response_model=IndexResponse)
def upload_and_index_pdf(file: UploadFile = File(...)):
    """
    CPU-bound parsing and embedding.
    Uses standard `def` so FastAPI offloads Docling chunking and indexing
    to a background worker threadpool instead of blocking the event loop.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Unique temp filename prevents collisions and strips any directory paths
    safe_name = os.path.basename(file.filename)
    temp_path = f"temp_{uuid.uuid4().hex}_{safe_name}"

    try:
        # Synchronous file write (runs safely inside FastAPI's threadpool)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Chunk and index
        docs = load_pdf(temp_path)
        bm25_index, texts, metadatas, ids, collection, embedding_model = hybrid_search(docs)

        # 2. Compile the headless LangGraph engine (no HITL pause for REST API)
        app.state.crag_app = build_crag_graph(
            bm25_index,
            texts,
            metadatas,
            ids,
            collection,
            embedding_model,
            enable_hitl=False,
        )

        return IndexResponse(
            message="PDF parsed, indexed, and LangGraph agent compiled.",
            chunks_processed=len(docs),
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse)
def query_crag(request: QueryRequest):
    """
    Synchronous execution (Cross-Encoder reranking + sync Groq SDK + MemorySaver).
    Kept as standard `def` so FastAPI runs it in a threadpool rather than
    freezing the ASGI event loop.
    """
    if app.state.crag_app is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF has been indexed yet. Please hit /upload_pdf first.",
        )

    # Every run with a checkpointer requires a thread_id in config
    thread_id = (request.thread_id or "").strip() or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": request.user_query,
        "documents": [],
        "retries": 0,
        "is_relevant": False,
        "generation": "",
        "final_context": "",
    }

    # .invoke() runs the graph straight to END and returns the final state dict
    final_state = app.state.crag_app.invoke(initial_state, config=config)

    is_relevant = bool(final_state.get("is_relevant", False))

    return QueryResponse(
        answer=final_state.get("generation") or "",
        context_used=(final_state.get("final_context") or "") if is_relevant else "",
        thread_id=thread_id,
    )