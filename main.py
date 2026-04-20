import os
from groq import Groq
from dotenv import load_dotenv
from embeddings import get_top_k

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# query = "Which framework is used for backend?"
query = "What is banana?"
top_docs = get_top_k(query, category="backend")

# 🔥 Step 2: Prepare context
context = "\n".join(top_docs)

# prompt = f"""
# Answer the question using ONLY the context below.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

prompt = f"""
You are a backend expert.

Use ONLY the provided context to answer.

If multiple frameworks are mentioned, list them clearly.
Do NOT mix features of different frameworks.

Context:
{context}

Question:
{query}

Answer:
"""
print("\nRetrieved Chunks:")
for doc in top_docs:
    print("-", doc)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
    temperature=0
)
print("\nFinal Answer:")
print(chat_completion.choices[0].message.content)