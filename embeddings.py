import numpy as np
import re
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-demo")

# Step 1: Documents
documents = [
    """FastAPI is a modern web framework built for building APIs with Python. 
    It is fast, easy to use, and uses Pydantic for validation. 
    It is built on Starlette and is widely used in backend systems.""",

    """Django is a high-level Python web framework that encourages rapid development. 
    It is used for backend development and comes with built-in features like ORM, authentication, and admin panel.""",

    """Python is widely used in artificial intelligence and machine learning. 
    It has powerful libraries like TensorFlow, PyTorch, and scikit-learn.""",

    """Bananas are rich in potassium and are good for health. 
    They are yellow in color and widely consumed across the world."""
]

# Step 2: Chunking
def chunk_text(text, chunk_size=3, overlap=1):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []

    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunk = sentences[start:end]
        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


# Step 3: Create chunks with metadata
all_chunks = []

for doc in documents:
    chunks = chunk_text(doc)

    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "category": "backend" if "framework" in chunk.lower() else "general"
        })

print("Number of chunks:", len(all_chunks))
print("Sample chunks:", all_chunks[:2])

# 🔥 Step 4: Create embeddings using Pinecone (FIXED)

vectors = []

for i, chunk in enumerate(all_chunks):
    embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[chunk["text"]],
        parameters={"input_type": "passage"}  # VERY IMPORTANT
    )[0]["values"]

    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {
            "text": chunk["text"],
            "category": chunk["category"]
        }
    })

# Step 5: Upload to Pinecone
index.upsert(vectors=vectors)

print("Uploaded to Pinecone:", len(vectors))


# 🔥 Step 6: Retrieval (UPDATED)

def get_top_k(query, k=3, category=None):

    # Query embedding (must match model)
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query"}  # IMPORTANT
    )[0]["values"]

    results = index.query(
        vector=query_embedding,
        top_k=k * 3,
        include_metadata=True,
        filter={"category": category} if category else None
    )

    final_results = []

    for match in results["matches"]:
        text = match["metadata"]["text"]

        # weak filtering
        if text.strip().startswith(("It", "They", "This")):
            continue

        final_results.append(text)

        if len(final_results) == k:
            break

    return final_results