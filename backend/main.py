from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from pathlib import Path
import torch
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from better_profanity import profanity

# Load default profanity words
profanity.load_censor_words()

# Optimization: Force PyTorch to use 1 thread to save RAM
torch.set_num_threads(1)

app = FastAPI(title="POI Search API - Balanced Mode")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MODEL_ID = "clip-ViT-B-32" # The most efficient CLIP model

# Mount static files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
qdrant_client = None
model = None

@app.on_event("startup")
async def startup_event():
    global qdrant_client, model
    
    # 1. Connect to Qdrant
    if QDRANT_URL:
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print("📡 Connected to Qdrant Cloud")
        except Exception as e:
            print(f"⚠️ Qdrant Error: {e}")

    # 2. Load Model Locally (Optimized for RAM)
    try:
        print(f"📥 Loading {MODEL_ID} locally...")
        model = SentenceTransformer(MODEL_ID, device="cpu")
        print("✅ Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")

def validate_query(text: str):
    """Enhanced validation for search queries to catch keyboard mashing."""
    text = text.strip()
    if not text:
        return False, "Query cannot be empty"
    
    if len(text) < 3:
        return False, "Query is too short (min 3 chars)"
    
    if text.isdigit():
        return False, "Query cannot be just numbers"

    # 1. Repetitive character check (e.g., "aaaaa")
    for char in set(text):
        if text.count(char) > 7 and char != ' ':
            return False, "Query contains too many repetitive characters"

    # 2. Word length check (e.g., "asdfghjklqwertyuiop")
    words = text.split()
    for word in words:
        if len(word) > 18:
            return False, "One of your words is unnaturally long"

    # 3. Vowel/Consonant Ratio (Keyboard mash check)
    vowels = set("aeiouAEIOU")
    v_count = sum(1 for c in text if c in vowels)
    alpha_count = sum(1 for c in text if c.isalpha())
    
    if alpha_count > 0:
        v_ratio = v_count / alpha_count
        # Natural languages usually have > 30% vowels. 
        if alpha_count > 6 and v_ratio < 0.25:
            return False, "Query looks like gibberish (low vowel ratio)"
            
        # Character diversity check (catch sadcasdasdasdas)
        unique_chars = len(set(text.lower().replace(" ", "")))
        if alpha_count > 8 and unique_chars / alpha_count < 0.45:
            return False, "Query uses too few unique characters (likely gibberish)"

        if alpha_count / len(text) < 0.4:
            return False, "Query contains too many non-alphabetic characters"
    
    # 4. Toxicity/Profanity Check
    if profanity.contains_profanity(text):
        return False, "Query contains inappropriate language"
        
    return True, text

@app.get("/")
async def root():
    return {
        "status": "Online", 
        "mode": "Local Optimized", 
        "qdrant": qdrant_client is not None,
        "model_loaded": model is not None
    }

@app.post("/search/text")
async def search_text(query: str, model_type: str = "base_clip", top_k: int = 5):
    if not model or not qdrant_client:
        raise HTTPException(status_code=500, detail="Server not fully initialized")
    
    is_valid, validated_query = validate_query(query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=validated_query)

    # Encode locally
    qv = model.encode(validated_query, convert_to_numpy=True).tolist()
    
    try:
        # Use query_search or simply search based on updated qdrant-client
        hits = qdrant_client.query_points(
            collection_name="poi_base_clip",
            query=qv,
            limit=top_k
        ).points
        
        # Filter results by a minimum similarity threshold (e.g. 0.28 raw score)
        # This prevents showing random people for weird inputs
        MIN_SCORE = 0.28
        valid_hits = [h for h in hits if h.score >= MIN_SCORE]

        if not valid_hits:
            return {
                "query": validated_query,
                "results": [],
                "message": "No confident matches found. Try described features like 'blonde hair' or 'wearing glasses'."
            }

        return {
            "query": validated_query, 
            "results": [
                {
                    "path": h.payload.get("path"), 
                    "score": min(0.99, (h.score * 2.5) + 0.1),
                    "raw_score": h.score
                } for h in valid_hits
            ]
        }
    except Exception as e:
        # Fallback to older search method if query_points fails
        try:
            hits = qdrant_client.search(
                collection_name="poi_base_clip",
                query_vector=qv,
                limit=top_k
            )
            return {
                "query": query, 
                "results": [
                    {
                        "path": h.payload.get("path"), 
                        "score": min(0.99, (h.score * 2.5) + 0.1),
                        "raw_score": h.score
                    } for h in hits
                ]
            }
        except Exception as e2:
            print(f"❌ Qdrant Query Error: {e2}")
            raise HTTPException(status_code=500, detail=str(e2))

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), model_type: str = "base_clip", top_k: int = 5):
    if not model or not qdrant_client:
        raise HTTPException(status_code=500, detail="Server not fully initialized")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Encode locally
    qv = model.encode(image, convert_to_numpy=True).tolist()
    
    try:
        hits = qdrant_client.query_points(
            collection_name="poi_base_clip",
            query=qv,
            limit=top_k
        ).points
        
        return {
            "results": [
                {
                    "path": h.payload.get("path"), 
                    "score": min(0.99, (h.score * 1.5)),
                    "raw_score": h.score
                } for h in hits
            ]
        }
    except Exception as e:
        # Fallback to older search method
        try:
            hits = qdrant_client.search(
                collection_name="poi_base_clip",
                query_vector=qv,
                limit=top_k
            )
            # handle hit scoring logic below...
        except Exception as e2:
             raise HTTPException(status_code=500, detail=str(e2))

        return {
            "results": [
                {
                    "path": h.payload.get("path"), 
                    "score": min(0.99, (h.score * 1.5)),
                    "raw_score": h.score
                } for h in hits
            ]
        }
