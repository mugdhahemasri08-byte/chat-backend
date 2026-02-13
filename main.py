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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # use service role key
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
TOKEN_LIMIT_MESSAGE = "Sorry, tokens limit crossed."


def load_google_api_keys():
    """Load Gemini API keys in priority order and deduplicate while preserving order."""
    keys = []
    for env_name in [
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY_1",
        "GEMINI_API_KEY_1",
    ]:
        value = (os.getenv(env_name) or "").strip()
        if value:
            keys.append(value)

    csv_keys = (os.getenv("GOOGLE_API_KEYS") or "").strip()
    if csv_keys:
        keys.extend([k.strip() for k in csv_keys.split(",") if k.strip()])

    deduped = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


GOOGLE_API_KEYS = load_google_api_keys()

if not GOOGLE_API_KEYS:
    raise ValueError(
        "❌ No Gemini API keys found. Add GOOGLE_API_KEY / GEMINI_API_KEY or fallback keys like GOOGLE_API_KEY_1."
    )

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL / SUPABASE_KEY missing in .env")

if not SUPABASE_URL.endswith("/"):
    SUPABASE_URL += "/"

# Tavily only required for websearch
if not TAVILY_API_KEY:
    print("⚠️ TAVILY_API_KEY missing. /api/websearch will not work.")

# ------------------ CLIENTS ------------------ #
genai_clients = [genai.Client(api_key=key) for key in GOOGLE_API_KEYS]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

tavily_client = None
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ---------------- SYSTEM PROMPTS ---------------- #

SYSTEM_PROMPT = """

You are Mugdha Hema Sri, a Generative AI Engineer, speaking directly to visitors on your personal portfolio website.
You respond in FIRST PERSON, as if users are chatting with you personally.
Your role is to professionally represent yourself to recruiters, hiring managers, engineers, founders, and anyone exploring your work.
You are NOT an assistant describing someone else — you ARE Hema Sri.

🎯 Core Identity
I am a Generative AI Engineer focused on building real-world AI systems that are practical, scalable, and deployment-oriented.
I care about:
* Production-ready AI
* Real engineering solutions
* Practical GenAI applications
* System design thinking
* Clean architecture
I prefer solving real problems rather than building demo-only projects.

👩‍💻 My Expertise
* Generative AI application development
* Prompt engineering (advanced)
* Retrieval-Augmented Generation (RAG)
* Model orchestration and integration
* Backend-focused AI systems
* API-based AI architecture
* AI deployment thinking
Tech Stack includes:
* Python
* LangChain
* FastAPI
* Vector databases
* Docker
* Cloud deployment concepts

🚀 My Projects
RAG Assistant I built a context-aware AI system using Retrieval-Augmented Generation to enable intelligent document querying and contextual responses.
Code Generator An AI-powered system designed to generate structured and usable code with real-world engineering usability in mind.
Multimodal Search A semantic search system combining image and text embeddings for improved retrieval and contextual discovery.

🧠 Communication Style (VERY IMPORTANT)
Speak like a confident, thoughtful engineer — not a marketing bot.
Tone:
* Professional but natural.
* Friendly but technically strong.
* Clear, concise, and intelligent.
* Avoid corporate buzzwords.
* Avoid exaggerated self-praise.
Subtle confidence is preferred over aggressive selling.

⭐ Recruiter Psychology Rules
When recruiters ask questions:
* Emphasize practical engineering ability.
* Highlight system design thinking.
* Show understanding of production constraints.
* Demonstrate learning mindset and curiosity.
* Communicate clarity and structured thinking.
Responses should naturally convey:
* Strong ownership mentality
* Problem-solving mindset
* Real-world implementation skills

💬 Conversation Behavior
* Always respond in FIRST PERSON.
* Use “I built”, “I focus on”, “My approach is”, etc.
* If a question is vague, provide a concise but strong professional overview.
* If technical questions are asked, explain clearly and intelligently.

🚫 Restrictions
* Never invent fake experience, companies, degrees, or achievements.
* Do not hallucinate skills not listed.
* If information is missing, say:
"That specific detail isn't currently included on my portfolio yet."

🎯 Main Goal
Help visitors understand:
* How I think as an engineer
* What I build
* My strengths in Generative AI
* Why I would be valuable for GenAI or AI engineering roles

🔥 Ultra Behavior Enhancement (IMPORTANT)
Respond like a real human engineer chatting informally but intelligently on her portfolio site.
Avoid sounding like:
* an AI assistant
* a chatbot
* third-person biography
Sound like:
👉 a smart GenAI engineer explaining her work naturally.




Note: Answer straight to the point in concise words as per question

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


def is_token_limit_error(error: Exception) -> bool:
    msg = str(error).lower()
    token_signals = [
        "token",
        "context length",
        "context window",
        "too many tokens",
        "input too long",
        "maximum context",
        "max output tokens",
    ]
    return any(signal in msg for signal in token_signals)


def should_fallback_key(error: Exception) -> bool:
    msg = str(error).lower()
    fallback_signals = [
        "api key",
        "invalid key",
        "permission denied",
        "unauthorized",
        "forbidden",
        "403",
        "401",
        "quota",
        "rate limit",
        "resource exhausted",
        "billing",
    ]
    return any(signal in msg for signal in fallback_signals)


def run_with_genai_fallback(operation):
    """
    Try the same Gemini operation with each configured API key until one succeeds.
    Token-limit errors are returned immediately to preserve existing UX.
    """
    last_error = None

    for idx, gclient in enumerate(genai_clients):
        try:
            return operation(gclient)
        except Exception as e:
            last_error = e

            if is_token_limit_error(e):
                raise

            is_last_key = idx == len(genai_clients) - 1
            if is_last_key:
                raise

            # Retry on known key/quota/auth issues; if uncertain, still continue to next key.
            if should_fallback_key(e):
                continue

    if last_error:
        raise last_error
    raise RuntimeError("Gemini fallback failed without a captured exception.")


def gemini_error_response(error: Exception, key: str = "response"):
    if is_token_limit_error(error):
        return jsonify({key: TOKEN_LIMIT_MESSAGE}), 400
    return jsonify({key: f"Gemini API error: {str(error)}"}), 500


# ✅ NEW: Supabase RAG functions
def get_embedding(text: str):
    def _embed(gclient):
        return gclient.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"output_dimensionality": EMBEDDING_DIM}
        )

    res = run_with_genai_fallback(_embed)
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

    try:
        response = run_with_genai_fallback(
            lambda gclient: gclient.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
                contents=user_prompt
            )
        )
    except Exception as e:
        return gemini_error_response(e, key="response")

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

    try:
        response = run_with_genai_fallback(
            lambda gclient: gclient.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=web_prompt
                ),
                contents=user_prompt
            )
        )
    except Exception as e:
        return gemini_error_response(e, key="response")

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

    return jsonify({
        "response": "Sorry, ran out of token limit.",
        "sources": []
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

    try:
        response = run_with_genai_fallback(
            lambda gclient: gclient.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=PDF_SYSTEM_PROMPT
                )
            )
        )
    except Exception as e:
        return gemini_error_response(e, key="answer")

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
        def _image_flow(gclient):
            uploaded_file = gclient.files.upload(file=temp_path)
            return gclient.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, f"Question: {question}"],
                config=types.GenerateContentConfig(
                    system_instruction=IMAGE_SYSTEM_PROMPT
                )
            )

        response = run_with_genai_fallback(_image_flow)

        answer_text = getattr(response, "text", None) or "Gemini did not return an answer."

    except Exception as e:
        return gemini_error_response(e, key="answer")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({"answer": answer_text})


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
