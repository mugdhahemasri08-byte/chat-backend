import os
from dotenv import load_dotenv
from supabase import create_client
from google import genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL.endswith("/"):
    SUPABASE_URL += "/"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GOOGLE_API_KEY)

EMBED_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"



# ---------- Embedding ----------
def get_embedding(text: str):
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return res.embeddings[0].values


# ---------- Search in Supabase ----------
def search_docs(question: str, top_k=5, threshold=0.15):
    q_emb = get_embedding(question)

    res = supabase.rpc(
        "match_documents",
        {
            "query_embedding": q_emb,
            "match_threshold": threshold,
            "match_count": top_k
        }
    ).execute()

    return res.data


# ---------- RAG Answer ----------
def rag_answer(question: str):
    matches = search_docs(question, top_k=5, threshold=0.15)

    if not matches:
        return "❌ No relevant document chunks found."

    # build context
    context_parts = []
    for m in matches:
        context_parts.append(
            f"[{m['file_name']} | chunk {m['chunk_index']} | sim={round(m['similarity'],3)}]\n{m['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not found in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return resp.text


# ---------- Run ----------
if __name__ == "__main__":
    print("✅ RAG Query Ready (type 'exit' to quit)")

    while True:
        q = input("\nAsk: ")
        if q.lower() in ["exit", "quit"]:
            break

        answer = rag_answer(q)
        print("\n🧠 Answer:\n", answer)
