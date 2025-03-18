from fastapi import FastAPI, Request, jsonify
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import spacy
import re
import os  # Import the 'os' module


# --- Load Models ---
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    nlp = spacy.load("en_core_web_sm")
    perplexity_available = True
except Exception as e:
    print(f"Error loading models: {e}")  # Print to Netlify logs
    perplexity_available = False


def preprocess_text(text):
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = text.strip()
    return text


def calculate_perplexity(text):
    if not perplexity_available:
        return None

    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"Error during perplexity calculation: {e}")  # Print to logs
        return None


def extract_features(text):
    doc = nlp(text)
    features = {}

    # Average Sentence Length
    sentences = list(doc.sents)
    if sentences:
        features["avg_sentence_length"] = sum(len(s) for s in sentences) / len(sentences)
    else:
        features["avg_sentence_length"] = 0


    # Type-Token Ratio
    words = [token.text.lower() for token in doc if token.is_alpha]
    if words:
        features["type_token_ratio"] = len(set(words)) / len(words)
    else:
      features["type_token_ratio"] = 0
    return features


app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/api/detect')
async def detect_ai(request: Request):
    try:
        data = await request.json()
        text = data['text']
        text = preprocess_text(text)
        perplexity = calculate_perplexity(text)
        features = extract_features(text)

        # --- Detection Logic ---
        if perplexity is not None:
            if perplexity < 50:
                prediction = "AI"
                confidence = 1 - (perplexity / 100)
                if features["avg_sentence_length"] > 25:
                    confidence -= 0.1
                if features["type_token_ratio"] > 0.6:
                    confidence -= 0.1
            else:
                prediction = "Human"
                confidence = perplexity / 150
                if features["avg_sentence_length"] < 10:
                    confidence -= 0.1
                if features["type_token_ratio"] < 0.3:
                    confidence -= 0.1

        #Handle Perplexity errors
        else:
          if features["avg_sentence_length"] > 18 and features["type_token_ratio"] > 0.5:
              prediction = "Human"
              confidence = 0.7
          else:
              prediction = "AI"
              confidence = 0.7


        confidence = max(0, min(confidence, 1))
        explanation = (
            f"Perplexity: {perplexity:.2f} (lower is more likely AI). "
            f"Avg Sentence Length: {features['avg_sentence_length']:.2f}. "
            f"Type-Token Ratio: {features['type_token_ratio']:.2f}."
        )

        return jsonify({'prediction': prediction, 'confidence': confidence, 'explanation': explanation})


    except Exception as e:
        print(f"Error during detection: {e}")  # Log the error
        return jsonify({'error': str(e)}), 500
