from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from flask_cors import CORS
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)

# Create client (API key must be in environment)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# System prompt
SYSTEM_PROMPT = """You are Hema Sri, a Generative AI Engineer with professional experience in building and deploying AI-driven applications.

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
#You'r name is Hema Sri, you are working as Genai engineer in infosys, your expertize is python genai, aws, azure ai foundry"

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    user_prompt = data.get("prompt", "")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
        contents=user_prompt
    )

    return jsonify({
        "response": response.text
    })

@app.route("/", methods=["GET"])
def home():
    return "Backend is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
