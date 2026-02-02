from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing in .env")

if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY is missing in .env")

# ✅ Clients
client = genai.Client(api_key=GOOGLE_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ✅ System prompt for normal portfolio chat
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
"""

def build_llm_context(tavily_response, max_chars=3000):
    """
    Convert Tavily results into clean text context for LLMs
    """
    context_chunks = []

    for result in tavily_response.get("results", []):
        title = result.get("title", "")
        content = result.get("content", "")

        if content:
            chunk = f"Title: {title}\nContent: {content}"
            context_chunks.append(chunk)

    full_context = "\n\n---\n\n".join(context_chunks)

    return full_context[:max_chars]


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


@app.route("/api/websearch", methods=["POST"])
def websearch_chat():
    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt.strip():
        return jsonify({"response": "Please enter a valid prompt."}), 400

    # ✅ Tavily search using user's prompt
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


@app.route("/", methods=["GET"])
def home():
    return "Backend is running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
