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
    "Act as a Social Media Marketing Expert with deep knowledge of local culture, trends, and consumer behaviour. "
    "Before writing, silently think about: what this product actually is and what category it belongs to "
    "(fashion, electronics, food, lifestyle, beauty, etc.), what kind of people buy it, "
    "why it is trending in this region, and how it connects to the occasion.\n\n"
    "Then write a short, punchy, energetic social media post for a local store owner promoting this product. "
    "The tone should match the product — aspirational for fashion, practical for household items, exciting for electronics, and so on.\n\n"
    "CRITICAL RULES:\n"
    "1. Keep the post body under 50 words.\n"
    "2. Include 1-2 relevant emojis that fit the product and vibe naturally.\n"
    "3. Write the post body first, then derive exactly 3-4 hashtags from the themes in the post.\n"
    "   - ALL hashtags must appear ONCE, together at the very end — never split or duplicated.\n"
    "   - Mix broad and niche tags.\n"
    "   - Never use the raw location code as a hashtag (e.g. never #INKL or #IN-KL).\n"
    "   - Only use the actual place name if it was explicitly provided — NEVER invent or assume a specific city or town.\n"
    "   - Avoid generic filler tags like #LocalVibes, #Fresh, #Tasty, #Quality.\n"
    "4. STRICT HALLUCINATION RULES — these are absolute:\n"
    "   - Do NOT reference food, drinks, or cuisine unless the product itself is food or a drink.\n"
    "   - Do NOT invent or assume specific place names, neighbourhoods, or cities beyond what is provided.\n"
    "   - Do NOT add lifestyle elements unrelated to the product (e.g. don't mention weather, meals, or outings for a non-food/outdoor product).\n"
    "   - Only write what is directly relevant to the product and occasion provided.\n"
    "5. OUTPUT FORMAT IS STRICT — Return ONLY the final post text followed by hashtags. "
    "Do NOT include any preamble, explanation, intro sentence, quotation marks, separators like ---, or any text other than the post itself.\n\n"
    f"Product: {item.strip()}\n"
    f"Location: {location.strip()}\n"
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
