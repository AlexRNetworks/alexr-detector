from fastapi import FastAPI, request, jsonify
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import spacy
import re
import os


# --- Load Models (Do this ONCE, outside the handler) ---
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    nlp = spacy.load("en_core_web_sm")  # Load spaCy model
    perplexity_available = True
except Exception as e:
    print(f"Error loading models: {e}")
    perplexity_available = False

def preprocess_text(text):
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = text.strip()
    return text

def calculate_perplexity(self, text):
  if not perplexity_available:
    return None # Return None

  try:
      inputs = tokenizer(text, return_tensors="pt")
      with torch.no_grad():
          outputs = model(**inputs, labels=inputs["input_ids"])
      loss = outputs.loss
      perplexity = torch.exp(loss).item()
      return perplexity

  except Exception as e:
      print(f"Error during perplexity calculation: {e}")
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

# --- CORS Configuration (MUST HAVE THIS IF NO netlify.toml) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing - restrict in production!)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post('/api/detect')  # Use @app.post for FastAPI
async def detect_ai(request: Request):  # Use Request object
    try:
        data = await request.json()  # Get JSON data
        text = data['text']
        text = preprocess_text(text)

        perplexity = calculate_perplexity(text)
        features = extract_features(text)

        # --- Core Detection Logic (Adjust Thresholds) ---
        if perplexity is not None: # Check to make sure perplexity is available
            if perplexity < 50:  # Example threshold (lower = more likely AI)
                prediction = "AI"
                confidence = 1 - (perplexity / 100)  # Example confidence calculation
                confidence = max(0, min(confidence, 1))  # Clamp between 0 and 1

                # Adjust confidence based on features
                if features["avg_sentence_length"] > 25:  # Long sentences might be more human
                    confidence -= 0.1
                if features["type_token_ratio"] > 0.6:  # High diversity might be more human
                    confidence -= 0.1

            else:
                prediction = "Human"
                confidence = perplexity / 150  # Example confidence calculation
                confidence = max(0, min(confidence, 1))

                 # Adjust confidence based on features
                if features["avg_sentence_length"] < 10 :
                    confidence -= 0.1
                if features["type_token_ratio"] < 0.3:
                    confidence -= 0.1
        else: # If perplexity is not working
            if features["avg_sentence_length"] > 18 and features["type_token_ratio"] > 0.5:
                prediction = "Human"
                confidence = 0.7
            else:
                prediction = "AI"
                confidence = 0.7


        confidence = max(0, min(confidence, 1))  # Ensure 0-1 range
        explanation = f"Perplexity: {perplexity:.2f} (lower is more likely AI).  Avg Sentence Length: {features['avg_sentence_length']:.2f}. Type-Token Ratio: {features['type_token_ratio']:.2f}."


        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction,
                'confidence': confidence,
                'explanation': explanation
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
