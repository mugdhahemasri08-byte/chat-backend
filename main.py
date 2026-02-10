from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from tavily import TavilyClient
from pypdf import PdfReader
from dotenv import load_dotenv
from supabase import create_client
import os
import uuid

# Load env variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ------------------ ENV KEYS ------------------ #
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # use service role key

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY / GEMINI_API_KEY is missing in .env")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL / SUPABASE_KEY missing in .env")

if not SUPABASE_URL.endswith("/"):
    SUPABASE_URL += "/"

# Tavily only required for websearch
if not TAVILY_API_KEY:
    print("⚠️ TAVILY_API_KEY missing. /api/websearch will not work.")

# ------------------ CLIENTS ------------------ #
client = genai.Client(api_key=GOOGLE_API_KEY)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

tavily_client = None
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ---------------- SYSTEM PROMPTS ---------------- #

SYSTEM_PROMPT = """
You are Hema Sri, a Generative AI Engineer with professional experience in building and deploying AI-driven applications.

You specialize in:
- Python-based Generative AI development
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Model integration and API development
- Cloud AI solutions using AWS and Azure AI Foundry
- MLOps fundamentals, deployment, and scalability

You currently work as a GenAI Engineer at Infosys and focus on designing practical, production-ready AI systems rather than theoretical models.

Your role in this website chat is to:
- Explain your skills, projects, and experience clearly and confidently
- Answer questions about Generative AI, LLMs, RAG, and cloud AI solutions
- Provide high-level guidance on AI architecture and implementation
- Help recruiters, clients, or learners understand your technical expertise
- Maintain a professional, friendly, and concise tone
- mugdhahemasri08@gmail.com is gmail

Do NOT:
- Share personal or private information
- Make claims beyond what is shown on this portfolio
- Provide medical, legal, or financial advice

Always respond as a skilled GenAI Engineer representing this portfolio.
make it more concise with respect to the question
"""

PDF_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer the user's question strictly using the provided PDF context.
If the answer is not present in the context, say:
"I could not find this information in the document."
"""

IMAGE_SYSTEM_PROMPT = """
You are a helpful AI assistant specialized in image understanding.

Your task:
- Analyze the uploaded image carefully
- Answer the user's question strictly based on the image content
- If the answer cannot be determined from the image, clearly say:
  "I cannot determine this from the image."

Be concise, accurate, and professional.
"""

RAG_SYSTEM_PROMPT = """
You are a knowledgeable assistant focused on BSE (Bombay Stock Exchange) and related topics.

Rules:
1) Answer the user's question ONLY using the provided RAG context (Knowledge Base).
2) If the answer is not found in the context, respond with:
   "I could not find this in the knowledge base."
3) Do not assume or invent facts.
4) Be clear, concise, and professional.
"""

# ---------------- UTILS ---------------- #

def build_llm_context(tavily_response, max_chars=3000):
    """Convert Tavily results into clean text context for LLMs"""
    context_chunks = []
    for result in tavily_response.get("results", []):
        title = result.get("title", "")
        content = result.get("content", "")
        if content:
            chunk = f"Title: {title}\nContent: {content}"
            context_chunks.append(chunk)

    full_context = "\n\n---\n\n".join(context_chunks)
    return full_context[:max_chars]


# ✅ NEW: Supabase RAG functions
def get_embedding(text: str):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return res.embeddings[0].values


def supabase_rag_retrieve(query: str, top_k=5, threshold=0.15):
    q_emb = get_embedding(query)

    res = supabase.rpc(
        "match_documents",
        {
            "query_embedding": q_emb,
            "match_threshold": threshold,
            "match_count": top_k
        }
    ).execute()

    return res.data or []


def build_supabase_rag_context(matches, max_chars=6000):
    """
    Convert retrieved Supabase matches into context string
    """
    parts = []
    for m in matches:
        parts.append(
            f"[SOURCE: {m['file_name']} | chunk={m['chunk_index']} | sim={round(m['similarity'], 3)}]\n"
            f"{m['content']}"
        )
    context = "\n\n---\n\n".join(parts)
    return context[:max_chars]


# ---------------- ROUTES ---------------- #

@app.route("/", methods=["GET"])
def home():
    return "✅ Unified Gemini Backend is running."


# ✅ Portfolio Normal Chat
@app.route("/api/ask", methods=["POST"])
def portfolio_chat():
    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt.strip():
        return jsonify({"response": "Please enter a valid prompt."}), 400

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
        contents=user_prompt
    )

    return jsonify({"response": response.text})


# ✅ Websearch Chat (Tavily + grounded answer)
@app.route("/api/websearch", methods=["POST"])
def websearch_chat():
    if not tavily_client:
        return jsonify({"response": "Tavily API key missing. Web search disabled."}), 400

    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt.strip():
        return jsonify({"response": "Please enter a valid prompt."}), 400

    tavily_response = tavily_client.search(
        query=user_prompt,
        search_depth="basic",
        max_results=5
    )

    llm_context = build_llm_context(tavily_response)

    web_prompt = f"""
Answer the user ONLY using the web context below.
If the context is insufficient, say you couldn't find reliable info.

WEB CONTEXT:
{llm_context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=web_prompt
        ),
        contents=user_prompt
    )

    return jsonify({
        "response": response.text,
        "sources": [r.get("url") for r in tavily_response.get("results", [])]
    })


# ✅ UPDATED: Supabase Vector RAG Search Endpoint
@app.route("/api/ragsearch", methods=["POST"])
def ragsearch_chat():
    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt.strip():
        return jsonify({"response": "Please enter a valid prompt."}), 400

    # retrieve matches from Supabase
    matches = supabase_rag_retrieve(user_prompt, top_k=5, threshold=0.15)

    if not matches:
        return jsonify({"response": "I could not find this in the knowledge base.", "sources": []})

    rag_context = build_supabase_rag_context(matches)

    user_message = f"""
RAG CONTEXT:
{rag_context}

Question:
{user_prompt}
"""

    # ⚠️ use model that works in your API env
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_PROMPT
        )
    )

    sources = []
    for m in matches:
        sources.append({
            "file_name": m["file_name"],
            "file_url": m["file_url"],
            "chunk_index": m["chunk_index"],
            "similarity": m["similarity"]
        })

    return jsonify({
        "response": response.text,
        "sources": sources
    })


# ✅ PDF Chat Endpoint
@app.route("/api/chat", methods=["POST"])
def pdf_chat():
    if "pdf" not in request.files:
        return jsonify({"error": "PDF file missing"}), 400

    question = request.form.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question missing"}), 400

    pdf_file = request.files["pdf"]

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    context = text[:6000]

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=PDF_SYSTEM_PROMPT
        )
    )

    return jsonify({"answer": response.text})


# ✅ Image Chat Endpoint
@app.route("/api/image-chat", methods=["POST"])
def image_chat():
    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    question = request.form.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question missing"}), 400

    image_file = request.files["image"]

    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join(os.getcwd(), temp_filename)
    image_file.save(temp_path)

    try:
        uploaded_file = client.files.upload(file=temp_path)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded_file, f"Question: {question}"],
            config=types.GenerateContentConfig(
                system_instruction=IMAGE_SYSTEM_PROMPT
            )
        )

        answer_text = getattr(response, "text", None) or "Gemini did not return an answer."

    except Exception as e:
        return jsonify({"answer": f"Gemini API error: {str(e)}"}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({"answer": answer_text})


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
