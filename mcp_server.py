import os
import uuid
from mcp.server.fastmcp import FastMCP

# Import your existing pipeline functions
# (Adjust these import names to match your actual files)
from loaders import load_pdf
from embeddings import hybrid_search
from langgraph_main import build_crag_graph

# Initialize the headless API server
mcp = FastMCP("DPDP_Legal_Agent")

# Global variable to hold our active compiled graph in RAM
active_crag_app = None

@mcp.tool()
def process_new_pdf(file_path: str) -> str:
    """
    Use this tool FIRST when the user wants to upload, read, or analyze a NEW legal document.
    It parses the PDF, builds the vector database, and initializes the RAG engine.
    """
    global active_crag_app
    
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
            
        # 1. Process the document
        chunks = load_pdf(file_path)
        
        # 2. Build the retrieval indices (BM25 + ChromaDB)
        bm25, texts, metadatas, ids, coll, emb = hybrid_search(chunks)
        
        # 3. Build the graph with HITL disabled for autonomous MCP execution
        active_crag_app = build_crag_graph(
            bm25, texts, metadatas, ids, coll, emb, enable_hitl=False
        )
        
        return f"Successfully processed {file_path}. The legal agent is ready for queries."
    
    except Exception as e:
        return f"Failed to process PDF: {str(e)}"

@mcp.tool()
def query_legal_database(question: str) -> str:
    """
    CRITICAL: ALWAYS use this tool to answer legal questions about the DPDP Act or uploaded contracts.
    You MUST run process_new_pdf first if a new document was just provided.
    """
    global active_crag_app
    
    if active_crag_app is None:
        return "Error: No legal document is currently loaded. Please ask the user to upload a PDF first."
        
    try:
        # Generate a fresh thread for this specific autonomous run
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "question": question,
            "documents": [],
            "retries": 0,
            "is_relevant": False,
            "generation": "",
            "final_context": ""
        }
        
        final_answer = ""
        
        # Execute the graph straight through (HITL is disabled)
        for step_output in active_crag_app.stream(initial_state, config=config):
            for node_name, state_update in step_output.items():
                if "generation" in state_update:
                    final_answer = state_update["generation"]
                    
        return final_answer or "Agent completed the pipeline but failed to generate a response."
        
    except Exception as e:
        return f"Error during graph execution: {str(e)}"

if __name__ == "__main__":
    # Bind the server to standard input/output for LLM client integration
    mcp.run(transport='stdio')