from flask import Flask, request, jsonify
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    prompt = data.get("prompt", "")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return jsonify({"response": response.text})


@app.route("/", methods=["GET"])
def home():
    return "Backend is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
