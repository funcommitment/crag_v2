import json
import time 
from retrieval import get_top_chunks
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq,RateLimitError
from typing import List,TypedDict,Dict,Any
from langgraph.graph import StateGraph,END 
from langgraph.checkpoint.memory import MemorySaver
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler


class Graphstate(TypedDict):
    question:str
    documents:List[Dict[str, Any]]
    retries:int
    generation:str
    is_relevant:bool
    final_context:str


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@observe(as_type="generation", name="groq_llm_call")
def call_llm(prompt: str,json_mode: bool = False,model_name: str = "openai/gpt-oss-120b",max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    # System prompt forcing JSON mode compliance if active
                    {"role": "system", "content": "You are a legal AI. Always output valid JSON if requested."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"} if json_mode else None,
                temperature=0.0,
            )
            if os.getenv("LANGFUSE_PUBLIC_KEY") and getattr(response, "usage", None):
                get_client().update_current_generation(
                    model=model_name,
                    usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    },
                )
            return response.choices[0].message.content

        except Exception as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "429" in str(e)

            if is_rate_limit and attempt < max_retries - 1:
                retry_after = None
                if getattr(e, "response", None) is not None:
                    retry_after = e.response.headers.get("retry-after")

                try:
                    wait_time = float(retry_after) + 1.5 if retry_after else 5 * (attempt + 1)
                except ValueError:
                    wait_time = 5 * (attempt + 1)

                print(f"[Rate Limit 429] Attempt {attempt + 1}/{max_retries}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            print(f"[API Error] {e}")
            # Failsafe: If the API crashes or exhausts retries, return safe fallback
            return '{"relevant": false}' if json_mode else "Error: Generation failed."

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

    compiled_graph = workflow.compile(checkpointer=memory,interrupt_before=["generator_node"] if enable_hitl else None)
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse_handler = CallbackHandler()
        compiled_graph = compiled_graph.with_config({"callbacks": [langfuse_handler]})

    return compiled_graph