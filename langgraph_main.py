import json
from retrieval import get_top_chunks
import os
from dotenv import load_dotenv
from groq import Groq
from typing import List,TypedDict,Dict,Any
from langgraph.graph import StateGraph,END 
from langgraph.checkpoint.memory import MemorySaver

class Graphstate(TypedDict):
    question:str
    documents:List[Dict[str, Any]]
    retries:int
    generation:str
    is_relevant:bool
    final_context:str

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str, json_mode: bool = False,model_name: str = "openai/gpt-oss-120b") -> str:
    try:
        response=client.chat.completions.create(
            model=model_name,
            messages=[
                    # System prompt forcing JSON mode compliance if active
                    {"role": "system", "content": "You are a legal AI. Always output valid JSON if requested."},
                    {"role": "user", "content": prompt}
                ],
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API Error] {e}")
        # Failsafe: If the API crashes, hand the while loop a safe 'False' to trigger a retry
        return '{"relevant": false}' if json_mode else "Error: Generation failed."

def build_crag_graph(bm25_index, texts, metadatas, ids, collection, embedding_model,enable_hitl=True):

    def retrieval_node(state:Graphstate):
        current_query=state["question"]
        results=get_top_chunks(query=current_query, 
                            bm25_index=bm25_index,
                            bm25_texts=texts,
                            bm25_metadatas=metadatas,
                            bm25_ids=ids,collection=collection,
                            embedding_model=embedding_model
                            ,top_k=3)
                            
        return {"documents":results}

    def grader_node(state:Graphstate):
        current_query=state["question"]
        top_chunks=state["documents"]

        context_blocks=[]
        for i,chunk in enumerate(top_chunks):
            context_blocks.append(f"--- Document {i+1} (Page {chunk['page']}) ---\n{chunk['text']}\n")
        final_context = "\n".join(context_blocks)
        grader_prompt = f"""You are a strict legal verification grader.
        Evaluate if the following context contains the exact factual answer to the question.
        If YES, output ONLY: {{"relevant": true}}
        If NO, output ONLY: {{"relevant": false}}
        
        Context:
        {final_context}
        
        Question: {current_query}"""

        grader_response = call_llm(grader_prompt, json_mode=True)
        try:
            is_relevant = json.loads(grader_response).get("relevant", False)
        except json.JSONDecodeError:
            is_relevant = False
        return {"is_relevant":is_relevant,"final_context":final_context}

    def hyde_rewrite_node(state:Graphstate):
        current_query=state["question"]
        current_retries=state["retries"]

        rewriter_prompt = f"""You are a legal research Rewriter.
        The original search query failed to find the specific legal statute.
        Rewrite the following question into a declarative statement that sounds like a legal clause, to improve vector search retrieval.
        Original Question: {current_query}"""
                        
        raw_rewrite = call_llm(rewriter_prompt,json_mode=False)

        return {"question":raw_rewrite,"retries":current_retries+1}

    def generator_node(state:Graphstate):
        current_query=state["question"]
        final_context=state["final_context"]

        generator_prompt = f"""You are an expert Indian Legal AI analyzing the DPDP Rules.
        Answer the question accurately using ONLY the provided context.
        
        Context:
        {final_context}
        
        Question: {current_query}"""
                    
        final_answer = call_llm(generator_prompt)
        return {"generation":final_answer}

    def failure_node(state: Graphstate):
        message = "I have searched the index 3 times, but this specific provision does not appear to be explicitly mentioned in the text. I cannot provide an answer without hallucinating."
        return {"generation": message}


    def route_after_grade(state:Graphstate):
        if state["is_relevant"]==True:
            return "generator_node"

        elif state["retries"]>=3:
            return "failure_node"

        else:
            return "hyde_rewrite_node"

    workflow=StateGraph(Graphstate)

    workflow.add_node("retrieval_node",retrieval_node)
    workflow.add_node("grader_node",grader_node)
    workflow.add_node("hyde_rewrite_node",hyde_rewrite_node)
    workflow.add_node("generator_node",generator_node)
    workflow.add_node("failure_node", failure_node)

    workflow.set_entry_point("retrieval_node")

    workflow.add_edge("retrieval_node", "grader_node")
    workflow.add_edge("hyde_rewrite_node","retrieval_node")
    workflow.add_edge("generator_node", END)
    workflow.add_edge("failure_node", END)

    workflow.add_conditional_edges(
        "grader_node",
        route_after_grade,
        {"generator_node":"generator_node",
        "hyde_rewrite_node":"hyde_rewrite_node",
        "failure_node": "failure_node"}
    )
    memory=MemorySaver()

    return workflow.compile(checkpointer=memory,interrupt_before=["generator_node"] if enable_hitl else [])