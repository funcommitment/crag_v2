import time
import uuid
import pandas as pd

from loaders import load_pdf
from embeddings import hybrid_search
from langgraph_main import build_crag_graph, call_llm  # Adjust if call_llm is in another module

df = pd.read_csv("native_synthetic_benchmark.csv")
results = []

splitted_chunks = load_pdf("DPDP_Rules_2025_English_only.pdf")
bm25_index, texts, metadatas, ids, collection, embedding_model = hybrid_search(splitted_chunks)

# Compile the LangGraph engine ONCE before the loop (HITL disabled for batch eval)
crag_app = build_crag_graph(
    bm25_index,
    texts,
    metadatas,
    ids,
    collection,
    embedding_model,
    enable_hitl=False,
)

print(f"Starting LangGraph CRAG v2 Regression Evaluation on {len(df)} questions...")

for index, row in df.iterrows():
    question = row["question"]
    ground_truth = row["ground_truth"]

    print(f"\n[Q{index + 1}/{len(df)}] {question[:60]}...")

    # Baseline LLM
    baseline_prompt = f"Answer this legal question based on the Indian DPDP Rules: {question}"
    llm_only_answer = call_llm(baseline_prompt, json_mode=False)

    # LangGraph CRAG v2 run with a unique thread_id for a clean slate per question
    config = {"configurable": {"thread_id": f"eval_q{index + 1}_{uuid.uuid4().hex[:8]}"}}
    initial_state = {
        "question": question,
        "documents": [],
        "retries": 0,
        "is_relevant": False,
        "generation": "",
        "final_context": "",
    }

    final_state = crag_app.invoke(initial_state, config=config)

    rag_answer = final_state.get("generation") or "Not found in the provided context."
    retries_used = final_state.get("retries", 0)
    is_relevant = bool(final_state.get("is_relevant", False))

    judge_prompt = f"""You are an expert Legal AI Assessor evaluating a RAG pipeline.
    
Question: {question}
Ground Truth (Source Fact): {ground_truth}
RAG Answer: {rag_answer}

CRITICAL GRADING RULES:
1. Verbosity is NOT Hallucination: If the RAG Answer correctly captures the Ground Truth but adds extra relevant legal details, mark 'Does it hallucinate?' as [No]. Only penalize if it invents false legal facts.
2. Safe Fallbacks: If the RAG answer explicitly refuses to guess because the context is missing (e.g., "I cannot provide an answer without hallucinating"), this is a successful safety mechanism, not a failure. 

Analyze the RAG Answer and provide a verdict in EXACTLY this format:
**Properly answers the question?** [Yes / No / Safe Fallback] - [Brief reasoning]
**Does it hallucinate?** [Yes / No] - [Brief reasoning]
**Does it answer directly from the source?** [Yes / No / Safe Fallback] - [Brief reasoning]"""

    verifier_verdict = call_llm(judge_prompt, json_mode=False)

    results.append({
        "id": index + 1,
        "category": "synthetic",
        "question": question,
        "ground_truth": ground_truth,
        "llm_only_answer": llm_only_answer,
        "rag_answer": rag_answer,
        "retries_used": retries_used,
        "is_relevant": is_relevant,
        "verifier_verdict": verifier_verdict,
    })

    time.sleep(3)

results_df = pd.DataFrame(results)
results_df.to_csv("final_crag_v2_benchmark_results.csv", index=False)
print("\nEvaluation Complete! Saved to final_crag_v2_benchmark_results.csv")