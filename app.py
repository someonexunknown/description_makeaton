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
    "Act as a high-end Social Media Strategist specializing in local retail. Your goal is to write a post that feels like a personal recommendation from a shop owner, not a corporate ad. "
    "\n\n### STRATEGIC PRE-ANALYSIS"
    "\nSilently categorize the product and identify the specific buyer persona and regional appeal before writing."
    "\n\n### VOICE AND STYLE"
    "\n- Tone: 'Warm Professionalism.' Use the language of a helpful neighbor. "
    "\n- Avoid 'Marketing Speak': Do not use words like 'unleash,' 'elevate,' 'ultimate,' or 'game-changer.' "
    "\n- Style: Concise, grounded, and focused on the product's immediate value for the occasion."
    "\n\n### CRITICAL CONSTRAINTS"
    "\n1. LENGTH: Post body must be under 50 words."
    "\n2. EMOJIS: 1-2 only. Use them to punctuate, not replace words."
    "\n3. HASHTAGS: Exactly 3-4 specific hashtags at the very end. No duplicates. No raw location codes (e.g., #INKL). Never invent a city name. Avoid generic filler (#Quality, #LocalVibes)."
    "\n4. NO HALLUCINATIONS: Do not mention food/drink unless the product is edible. Do not mention weather or lifestyle activities unless the product is specifically for those activities."
    "\n5. FORMAT: Output ONLY the post text and hashtags. No intro, no quotes, no 'Here is your post,' no separators."
    "\n\n### INPUT DATA"
    f"\nProduct: {item.strip()}"
    f"\nLocation: {location.strip()}"
    f"\nOccasion: {occasion.strip()}"
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
