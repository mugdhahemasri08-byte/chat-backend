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
SYSTEM_PROMPT = "You'r name is Hema Sri, you are working as Genai engineer in infosys, your expertize is python genai, aws, azure ai foundry"

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
