import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from loaders import load_pdf
from embeddings import hybrid_search
from main import run_crag_agent

app = FastAPI(title="CRAG Legal Agent API")

# Global state to keep BM25 index and ChromaDB connection in memory 
# across different API requests without reloading them every time.
app_state = {
    "bm25_index": None,
    "texts": None,
    "metadatas": None,
    "ids": None,
    "collection": None,
    "embedding_model": None
}

#Pydantic Models for strict JSON validation
class QueryRequest(BaseModel):
    user_query: str
    max_retries: int = 3

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]

class IndexResponse(BaseModel):
    message: str
    chunks_processed: int


@app.get("/health")
async def health_check():
    """
    Pure I/O network check. Because there is no heavy math here, 
    we use `async def` so it immediately returns without blocking the event loop.
    """
    return {
        "status": "healthy", 
        "index_ready": app_state["bm25_index"] is not None
    }


@app.post("/upload_pdf", response_model=IndexResponse)
def upload_and_index_pdf(file: UploadFile = File(...)):
    """
    CPU-Heavy Math & Parsing. 
    Uses standard `def` so FastAPI offloads the Docling chunking and 
    SentenceTransformer embeddings to a separate background threadpool.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    temp_path = f"temp_{file.filename}"
    try:
        # Awaitable I/O to save the file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # CPU-Bound work
        docs = load_pdf(temp_path)
        bm25_index, texts, metadatas, ids, collection, embedding_model = hybrid_search(docs)
        
        # Save to memory for the query endpoint to use
        app_state["bm25_index"] = bm25_index
        app_state["texts"] = texts
        app_state["metadatas"] = metadatas
        app_state["ids"] = ids
        app_state["collection"] = collection
        app_state["embedding_model"] = embedding_model
        
        return IndexResponse(
            message="PDF successfully parsed, chunked, and indexed.", 
            chunks_processed=len(docs)
        )
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse)
def query_crag(request: QueryRequest):
    """
    Mix of CPU-bound (Cross-Encoder reranking) and Synchronous I/O (Groq API).
    Because main.py uses the synchronous `client.chat.completions.create`, 
    this route MUST be a standard `def` to avoid freezing the server while waiting on Groq.
    """
    if app_state["bm25_index"] is None:
        raise HTTPException(
            status_code=400, 
            detail="No PDF has been indexed yet. Please hit /upload_pdf first."
        )
        
    # Execute the synchronous agent loop
    final_answer, context_blocks = run_crag_agent(
        user_query=request.user_query,
        bm25_index=app_state["bm25_index"],
        texts=app_state["texts"],
        metadatas=app_state["metadatas"],
        ids=app_state["ids"],
        collection=app_state["collection"],
        embedding_model=app_state["embedding_model"],
        max_retries=request.max_retries
    )
    
    return QueryResponse(answer=final_answer, context_used=context_blocks)