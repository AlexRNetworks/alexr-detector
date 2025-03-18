from fastapi import FastAPI, Request, jsonify
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- VERY Simple AI Word List (Expand this!) ---
ai_words = ["generated", "artificial", "intelligence", "algorithm", "model", "neural", "network", "machine", "learning"]

@app.post("/api/detect")
async def detect_ai(request: Request):
    try:
        data = await request.json()
        text = data['text'].lower()  # Convert to lowercase for case-insensitivity
        words = text.split()
        ai_word_count = 0

        for word in words:
            if word in ai_words:
                ai_word_count += 1

        confidence = min(1.0, ai_word_count / len(words) if len(words) >0 else 0)  # Basic confidence calculation
        prediction = "AI" if confidence >= 0.4 else "Human" # Keep it simple
        explanation = f"Based on a simple keyword check. AI word count: {ai_word_count}"
        if len(words) == 0:
            confidence = 0;
            prediction = "Human"
            explanation = "Enter Text"

        return jsonify({'prediction': prediction, 'confidence': confidence, 'explanation': explanation})

    except Exception as e:
        print(f"Error: {e}")  # Log errors to Netlify Function logs
        return jsonify({'error': str(e)}), 500
