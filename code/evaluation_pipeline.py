import time
import json
import argparse

from loader import load_json, extract_turns, extract_contexts
from metrices import compute_relevance, hallucinations, latency_cost_evaluation


def evaluate(chat_path, context_path):
    # Load JSON files
    chat_json = load_json(chat_path)
    context_json = load_json(context_path)

    # Extract data
    user_query, assistant_response = extract_turns(chat_json)
    contexts = extract_contexts(context_json)

    # Start timer
    start_time = time.time()

    # Run metrics
    relevance_result = compute_relevance(
        user_query, assistant_response, contexts
    )
    factual_result = hallucinations(
        assistant_response, contexts
    )

    # Estimate tokens
    token_count = len((user_query + assistant_response).split())

    # End timer
    end_time = time.time()
    performance_result = latency_cost_evaluation(
        start_time, end_time, token_count
    )

    # Combine results
    final_result = {
        "relevance": relevance_result,
        "factual accuracy": factual_result,
        "performance": performance_result
    }

    return final_result


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument("--chat", required=True, help="Path to chat JSON")
    parser.add_argument("--context", required=True, help="Path to context JSON")
    parser.add_argument("--output", required=False, help="Path to save output JSON")

    args = parser.parse_args()

    result = evaluate(args.chat, args.context)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("✅ Evaluation result saved to:", args.output)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
