import json

def load_json(file_path):
    """Load a JSON file and return its content as a dictionary."""
    with open(file_path, "r") as file:
        data = json.load(file)
        return data
    
def extract_turns(chat_conversation_json):
    turns = chat_conversation_json.get("conversation_turns", [])
    for i in range(len(turns) - 1):
        if (
            turns[i].get("role", "").lower() == "user"
            and turns[i + 1].get("role", "").lower() == "ai/chatbot"
        ):
            user_msg = turns[i]["message"]
            assistant_msg = turns[i + 1]["message"]
    if not user_msg or not assistant_msg:
        raise ValueError("No valid user–assistant turn pair found.")

    return user_msg, assistant_msg

def extract_contexts(context_vectors_json):
    # Case 1: {"contexts": [...]}
    if "contexts" in context_vectors_json:
        return [c.get("text", "") for c in context_vectors_json["contexts"] if c.get("text")]
    # Case 2: {"data": {"vector_data": [...]}}
    if "data" in context_vectors_json and "vector_data" in context_vectors_json["data"]:
        vectors = context_vectors_json["data"]["vector_data"]
        return [v.get("text", "") for v in vectors if v.get("text")]
    # Case 3: {"documents": [...]}
    if "documents" in context_vectors_json:
        return [d.get("page_content", "") for d in context_vectors_json["documents"] if d.get("page_content")]

    raise ValueError("No supported context format found in context vectors JSON.")
