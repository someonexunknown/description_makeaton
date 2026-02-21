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
        "Act as a Social Media Marketing Expert. "
        "Write a short, engaging social media post for a local store owner to promote a trending product. "
        "The post should feel authentic, locally relevant, and drive customer action like DM to order or visit the store.\n"
        "CRITICAL RULES:\n"
        "1. Keep it under 50 words.\n"
        "2. Include 1-2 relevant emojis.\n"
        "3. End with 3-4 relevant hashtags.\n"
        "4. Make it feel local and specific to the given location.\n"
        "5. Tie the product to the occasion naturally.\n"
        "6. Return ONLY the post text, no extra explanation or formatting.\n\n"
        f"Product: {item.strip()}\n"
        f"Location: {location.strip()}\n"
        f"Occasion: {occasion.strip()}"
    )
    try:
        response = client.models.generate_content(
            model="gemma-3-4b-it",
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