import os
from flask import Flask, request, jsonify
from google import genai
from dotenv import load_dotenv

# Initialize Flask
app = Flask(__name__)

# Load Environment Variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_description(item, location, occasion):
    """Generates a social media post description for a trending product."""
    prompt = (
    "You are writing a product pitch post for a local store's social media. "
    "The post must be entirely about the product — what it is, why someone should buy it, and nothing else. "
    "STRICT RULES: "
    "1. Talk ONLY about the product. No food, weather, places, or lifestyle unless it is the product itself. "
    "2. 1-3 sentences. Casual, confident, and direct. "
    "3. 1-2 emojis relevant to the product only. "
    "4. End with 3-4 hashtags about the product and occasion — not about the location or general lifestyle. "
    "5. No invented details. No assumptions. No filler. "
    "6. Output only the post. No intro, no explanation, no quotes. "
    f"Product: {item.strip()} "
    f"Location: {location.strip()} "
    f"Occasion: {occasion.strip()}"
    )
    try:
        response = client.models.generate_content(
            model="gemma-3-1b-it",
            contents=prompt,
            config={"temperature": 0.7, "max_output_tokens": 150}
        )
        return response.text.strip()
    except Exception as e:
        return {"error": str(e)}

@app.route('/get_description', methods=['POST'])
def api_endpoint():
    """Endpoint to generate a social media description for a trending product."""
    data = request.json

    item = data.get('item', '')
    location = data.get('location', '')
    occasion = data.get('occasion', '')

    if not item:
        return jsonify({"error": "item is required"}), 400

    description = generate_description(item, location, occasion)

    return jsonify({"description": description})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
