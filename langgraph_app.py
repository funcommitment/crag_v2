import streamlit as st
import tempfile
import os
import uuid

# --- INJECT OFFLINE FLAGS BEFORE ANY OTHER IMPORTS ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from loaders import load_pdf
from embeddings import hybrid_search
from langgraph_main import build_crag_graph 

st.set_page_config(page_title="CRAG Legal Verification", page_icon="⚖️", layout="centered")
st.title("⚖️ CRAG DPDP Act AI Agent")

# 1. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph_paused" not in st.session_state:
    st.session_state.graph_paused = False
if "pending_context" not in st.session_state:
    st.session_state.pending_context = ""

# 2. Sidebar for Dynamic Data Ingestion
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")
    
    if uploaded_file:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.status("Initializing Production Pipeline...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                status.write("Extracting and chunking PDF text...")
                splitted_chunks = load_pdf(tmp_path)
                os.remove(tmp_path) 
                
                status.write("Vectorizing & building Hybrid Search...")
                bm25_index, texts, metadatas, ids, collection, embedding_model = hybrid_search(splitted_chunks)
                
                status.write("Compiling LangGraph State Machine...")
                st.session_state.crag_app = build_crag_graph(
                    bm25_index, texts, metadatas, ids, collection, embedding_model
                )
                
                st.session_state.db_ready = True
                st.session_state.current_file = uploaded_file.name 
                status.update(label="Pipeline Ready!", state="complete", expanded=False)

# 3. Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle Paused Graph Resumption (Human-in-the-Loop & Time Travel)
if st.session_state.graph_paused:
    st.warning("⚠️ **Human-in-the-Loop Interrupt:** Review or edit the retrieved legal context before the AI generates its answer.")
    
    # TIME TRAVEL: Allow the human to actually edit the state memory
    edited_context = st.text_area(
        "📄 Edit Context (Modifying this alters the graph's memory)", 
        value=st.session_state.pending_context, 
        height=300
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Generate Response", use_container_width=True):
            with st.chat_message("assistant"):
                with st.spinner("Generating final response..."):
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    # If the human changed the text, forcefully update the graph's state (Time Travel)
                    if edited_context != st.session_state.pending_context:
                        st.session_state.crag_app.update_state(config, {"final_context": edited_context})
                    
                    final_answer = ""
                    # Resume graph by passing None
                    for step_output in st.session_state.crag_app.stream(None, config=config):
                        for node_name, state_update in step_output.items():
                            if "generation" in state_update:
                                final_answer = state_update["generation"]
                    
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    
                    # RESET: Cycle the thread ID so the next question doesn't collide with this one
                    st.session_state.graph_paused = False
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.rerun()

    with col2:
        if st.button("❌ Reject & Cancel", type="primary", use_container_width=True):
            # Gracefully log the rejection
            rejection_msg = "*(Agent execution halted by human reviewer)*"
            st.session_state.messages.append({"role": "assistant", "content": rejection_msg})
            
            # RESET: Cycle the thread ID to cleanly sever the paused graph
            st.session_state.graph_paused = False
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
            
    st.stop() # Halts script so the chat input below doesn't render while paused

# 5. User Interaction & Graph Execution (Initial Run)
if user_query := st.chat_input("Ask a question regarding the uploaded rules..."):
    if "db_ready" not in st.session_state:
        st.error("Please upload a PDF in the sidebar to initialize the databases.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("🧠 Agent State Machine Booting...", expanded=True) as status:
            initial_state = {
                "question": user_query,
                "documents": [],
                "retries": 0,
                "is_relevant": False,
                "generation": "",
                "final_context": ""
            }
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            final_answer = ""
            
            for step_output in st.session_state.crag_app.stream(initial_state, config=config):
                for node_name, state_update in step_output.items():
                    status.write(f"⚙️ **Finished Node:** `{node_name}`")
                    
                    if node_name == "retrieval_node":
                        docs = state_update.get("documents", [])
                        status.write(f"🔍 *Retrieved {len(docs)} chunks from Vector DB.*")
                        with status.expander("View Retrieved Documents"):
                            for idx, doc in enumerate(docs):
                                st.markdown(f"**Document {idx+1} (Page {doc.get('page')}):**\n\n{doc.get('text')}")
                                st.divider()
                                
                    elif node_name == "grader_node":
                        is_rel = state_update.get("is_relevant")
                        st.session_state.pending_context = state_update.get("final_context", "")
                        if is_rel:
                            status.write("✅ *Grader Result: Context is RELEVANT.*")
                        else:
                            status.write("❌ *Grader Result: Context is IRRELEVANT. Triggering fallback.*")
                            
                    elif node_name == "hyde_rewrite_node":
                        new_q = state_update.get("question", "")
                        retries = state_update.get("retries", 0)
                        status.write(f"🔄 *HyDE Rewrite (Attempt {retries}):* {new_q}")
                        
                    elif node_name == "failure_node":
                        status.write("⚠️ *Hit maximum retries (3). Forcing graceful failure.*")
                        
                    if "generation" in state_update:
                        final_answer = state_update["generation"]

            status.update(label="Graph Execution Paused / Complete", state="complete", expanded=False)
            
        # The ultimate check: Did it generate an answer (failure node) or did it hit the HITL interrupt?
        if not final_answer:
            st.session_state.graph_paused = True
            st.rerun()
        else:
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            # Cycle thread ID here too if the failure node executed, preventing collisions
            st.session_state.thread_id = str(uuid.uuid4())