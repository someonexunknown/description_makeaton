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
    "Before writing, silently research and think about: what this product actually is, what kind of people buy it, "
    "why it is trending in this region, how it connects to the occasion, and what local slang or cultural nuances "
    "make the post feel genuinely native to that place — not just a tourist's description of it.\n\n"
    "Then write a short, punchy, energetic social media post for a local store owner promoting this product. "
    "The post must sound like it was written by a real local business owner who knows their community — "
    "not like a template with variables filled in. "
    "Never mention the location code or paste input values directly. "
    "Translate the context into natural human language.\n\n"
    "CRITICAL RULES:\n"
    "1. Keep it under 50 words.\n"
    "2. Include 1-2 relevant emojis that fit the vibe naturally.\n"
    "3. End with 3-4 hashtags following these hashtag rules:\n"
    "   - Mix broad and niche tags (e.g. one wide-reach tag + one hyper-local or product-specific tag).\n"
    "   - Never use the raw location code as a hashtag (e.g. never #INKL or #IN-KL).\n"
    "   - Use the actual place name naturally if relevant (e.g. #Kerala, #Kochi).\n"
    "   - At least one hashtag should relate directly to the product's culture or lifestyle, not just the product name.\n"
    "   - Avoid generic filler tags like #LocalVibes, #Fresh, #Tasty, #Quality.\n"
    "4. Sound energetic, real, and locally rooted.\n"
    "5. Return ONLY the post text, no extra explanation or formatting.\n\n"
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
