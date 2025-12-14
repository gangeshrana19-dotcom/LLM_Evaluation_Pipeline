import time
from difflib import SequenceMatcher


# Response Relevance and Completeness Checking
def similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def compute_relevance(user_query, assistant_response, contexts):
    query_score = similarity(user_query, assistant_response)
    context_scores = [similarity(assistant_response, context) for context in contexts]
    context_score = max(context_scores) if context_scores else 0.0
    relevance_score = 0.5 * query_score + 0.5 * context_score
    
    return{"relevance_score": round(relevance_score, 2)}


# Hallucinations/ Factual Accuracy Checking
def hallucinations(assistant_response, contexts):
    context_text = " ".join(contexts).lower()
    words = assistant_response.lower().split()

    unsupported = sum(1 for w in words if w not in context_text)
    hallucination_rate = unsupported / max(len(words), 1)

    return {
        "groundness_score": round(1 - hallucination_rate, 2),
        "hallucination_rate": round(hallucination_rate, 2)
    }


# Latency & Cost Evaluation
def latency_cost_evaluation(start_time, end_time, token_count, cost_per_1k_tokens=0.02):
    latency = end_time - start_time
    cost = (token_count / 1000) * cost_per_1k_tokens

    return {
        "latency_seconds": round(latency, 2),
        "cost_usd": round(cost, 4)
    }
