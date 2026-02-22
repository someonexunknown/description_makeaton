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
        "Write a social media post for a local store owner selling this product. "
        "The post must sound human, grounded, and natural — not like AI wrote it. "
        "Understand what the product is, who buys it, and why it matters to them before writing. "
        "STRICT RULES: "
        "1. 1-3 sentences max. Casual but credible tone. "
        "2. 1-2 emojis that feel natural to the product, not forced. "
        "3. End with 3-4 hashtags that echo the words and feeling of the post. "
        "4. Use the real location name naturally in the post or hashtag if it adds value — never paste location codes. "
        "5. Stick strictly to the product — no invented food, places, weather, or lifestyle references. "
        "6. Output only the post. No intro, no explanation, no quotes around it. "
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
