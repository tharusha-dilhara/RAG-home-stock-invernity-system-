from flask import Flask, request, jsonify
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_nvidia_ai_endpoints import ChatNVIDIA

app = Flask(__name__)

# Initialize ChatNVIDIA client (replace with your actual API key)
client = ChatNVIDIA(
    model="qwen/qwen2.5-7b-instruct",
    api_key="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",  # Replace with your API key
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Example corpus: a list of documents
documents = documents = [
    "January: Home Stock Inventory: Batteries: 20: $1.50",
    "January: Home Stock Inventory: Light Bulbs: 15: $2.00",
    "January: Groceries: Milk: 10: $3.50",
    "January: Groceries: Bread: 8: $2.50",
    "January: Home Essentials: Toilet Paper: 12: $0.99",
    "January: Home Essentials: Cleaning Spray: 6: $4.00",
    "February: Home Stock Inventory: Batteries: 18: $1.55",
    "February: Home Stock Inventory: Light Bulbs: 14: $2.05",
    "February: Groceries: Milk: 12: $3.60",
    "February: Groceries: Bread: 7: $2.55",
    "February: Home Essentials: Toilet Paper: 10: $1.00",
    "February: Home Essentials: Cleaning Spray: 7: $4.10",
    "March: Home Stock Inventory: Batteries: 22: $1.45",
    "March: Home Stock Inventory: Light Bulbs: 16: $1.95",
    "March: Groceries: Milk: 11: $3.55",
    "March: Groceries: Bread: 9: $2.50",
    "March: Home Essentials: Toilet Paper: 11: $1.02",
    "March: Home Essentials: Cleaning Spray: 5: $4.05",
    "April: Home Stock Inventory: Batteries: 19: $1.50",
    "April: Home Stock Inventory: Light Bulbs: 17: $2.00",
    "April: Groceries: Milk: 13: $3.70",
    "April: Groceries: Bread: 10: $2.60",
    "April: Home Essentials: Toilet Paper: 12: $1.00",
    "April: Home Essentials: Cleaning Spray: 6: $4.15",
    "May: Home Stock Inventory: Batteries: 21: $1.52",
    "May: Home Stock Inventory: Light Bulbs: 18: $2.02",
    "May: Groceries: Milk: 14: $3.75",
    "May: Groceries: Bread: 11: $2.65",
    "May: Home Essentials: Toilet Paper: 13: $1.03",
    "May: Home Essentials: Cleaning Spray: 7: $4.20"
]

# Compute vector embeddings for each document
doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

# Build a FAISS index for similarity search
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def retrieve(query, top_k=3):
    """
    Retrieve the top_k most relevant documents for the given query.
    """
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec, dtype=np.float32)
    distances, indices = index.search(query_vec, top_k)
    return [documents[idx] for idx in indices[0]]

def rag_query(query):
    """
    Build the prompt by retrieving context documents and generating an answer.
    """
    retrieved_docs = retrieve(query)
    prompt = "Answer the following question using the provided context.\n\nContext:\n"
    for doc in retrieved_docs:
        prompt += f"- {doc}\n"
    prompt += f"\nQuestion: {query}\n\nAnswer:"
    response = client.invoke([{"role": "user", "content": prompt}])
    return response.content

@app.route('/rag', methods=['POST'])
def get_rag_answer():
    """
    Flask endpoint that accepts a JSON payload with a 'query' key
    and returns the generated answer.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Please provide a query in the JSON body with key 'query'."}), 400

    query = data['query']
    answer = rag_query(query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
