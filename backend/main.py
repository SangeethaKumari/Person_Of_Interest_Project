from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import requests
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from better_profanity import profanity

# Safety filters
profanity.load_censor_words()

app = FastAPI(title="POI Search API - Final Cloud Mode")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "openai/clip-vit-base-patch32"

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

# Global client
qdrant_client = None

@app.on_event("startup")
async def startup_event():
    global qdrant_client
    if QDRANT_URL:
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print("📡 Connected to Qdrant Cloud")
        except Exception as e:
            print(f"❌ Qdrant Error: {e}")

def validate_query(text: str):
    text = text.strip()
    if not text or len(text) < 3:
        return False, "Query too short or empty"
    if profanity.contains_profanity(text):
        return False, "Query contains inappropriate language"
    
    # Gibberish check
    vowels = set("aeiouAEIOU")
    v_count = sum(1 for c in text if c in vowels)
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count > 6 and (v_count / alpha_count) < 0.20:
        return False, "Query looks like gibberish"
        
    return True, text

def get_hf_embeddings(inputs, is_image=False):
    """The 'Proxy' method: Offload math to HF to save 100% of Render RAM."""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN missing in environment")
    
    # Use the official v1 embeddings endpoint (most stable)
    API_URL = "https://router.huggingface.co/hf-inference/v1/embeddings"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        if is_image:
            # For images we use the task-specific endpoint
            IMAGE_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
            response = requests.post(IMAGE_URL, headers=headers, data=inputs)
        else:
            # Use OpenAI compatible format for text
            response = requests.post(API_URL, headers=headers, json={
                "model": MODEL_ID,
                "input": inputs
            })
            
        if response.status_code != 200:
            print(f"HF Error Status: {response.status_code}")
            print(f"HF Error Text: {response.text}")
            raise HTTPException(status_code=500, detail="AI Engine is busy/denied. Please check HF_TOKEN permissions.")
            
        data = response.json()
        
        # Parse based on endpoint format
        if "data" in data: # OpenAI format
            return data["data"][0]["embedding"]
        elif isinstance(data, list):
            return data[0] if isinstance(data[0], list) else data
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "Online", 
        "mode": "Zero-RAM Cloud Proxy", 
        "env_check": {
            "hf_token_present": HF_TOKEN is not None and len(str(HF_TOKEN)) > 5,
            "qdrant_url_present": QDRANT_URL is not None,
            "qdrant_api_key_present": QDRANT_API_KEY is not None
        }
    }

@app.post("/search/text")
async def search_text(query: str, model_type: str = "base_clip", top_k: int = 5):
    is_valid, v_query = validate_query(query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=v_query)

    v_math = get_hf_embeddings(v_query)
    
    try:
        hits = qdrant_client.query_points(
            collection_name="poi_base_clip",
            query=v_math,
            limit=top_k
        ).points
        
        MIN_SCORE = 0.28
        results = [
            {
                "path": h.payload.get("path"), 
                "score": min(0.99, (h.score * 2.5) + 0.1),
                "raw_score": h.score
            } for h in hits if h.score >= MIN_SCORE
        ]
        return {"query": v_query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), model_type: str = "base_clip", top_k: int = 5):
    contents = await file.read()
    v_math = get_hf_embeddings(contents, is_image=True)
    
    try:
        hits = qdrant_client.query_points(
            collection_name="poi_base_clip",
            query=v_math,
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
        raise HTTPException(status_code=500, detail=str(e))
