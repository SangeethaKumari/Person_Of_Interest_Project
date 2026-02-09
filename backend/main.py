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

app = FastAPI(title="POI Search API - Smart Hybrid Mode")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sentence-transformers/clip-ViT-B-32"

# Detection: Are we on Render or Local Mac?
IS_RENDER = os.getenv("RENDER") is not None

# Global clients
qdrant_client = None
local_model = None

@app.on_event("startup")
async def startup_event():
    global qdrant_client, local_model
    
    # 1. Database Connection
    if QDRANT_URL:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("📡 Connected to Qdrant Cloud")

    # 2. Intelligence Selection
    if not IS_RENDER:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            torch.set_num_threads(1)
            print(f"📥 [LOCAL MODE] Loading {MODEL_ID}...")
            local_model = SentenceTransformer(MODEL_ID, device="cpu")
            print("✅ Local Model Ready")
        except Exception as e:
            print(f"⚠️ Local Model failed, falling back to Cloud: {e}")
    else:
        print("☁️ [RENDER MODE] Using Zero-RAM Cloud Proxy")

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

def get_embeddings(inputs, is_image=False):
    """Smart Hybrid: Uses Local if available, else Cloud."""
    if local_model:
        # Use your Mac's power
        return local_model.encode(inputs, convert_to_numpy=True).tolist()
    
    # Fallback to Cloud (Zero RAM for Render)
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN missing")
    
    # Use the 2026 Router with explicit task mapping
    API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "X-Wait-For-Model": "true"  # Crucial: forces HF to load the model if it's cold
    }
    
    try:
        if is_image:
            response = requests.post(API_URL, headers=headers, data=inputs)
        else:
            response = requests.post(API_URL, headers=headers, json={"inputs": inputs})
            
        if response.status_code != 200:
            print(f"HF Error: {response.status_code} - {response.text}")
            msg = "AI Service Busy (Loading...)" if response.status_code == 503 else "AI Access Denied"
            raise HTTPException(status_code=500, detail=f"{msg}. Status: {response.status_code}")
            
        data = response.json()
        return data[0] if isinstance(data, list) and isinstance(data[0], list) else data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "status": "Online", 
        "mode": "Hybrid", 
        "engine": "Local" if local_model else "Cloud Proxy",
        "render_detected": IS_RENDER
    }

@app.post("/search/text")
async def search_text(query: str, model_type: str = "base_clip", top_k: int = 5):
    if profanity.contains_profanity(query):
        raise HTTPException(status_code=400, detail="Inappropriate language")

    v_math = get_embeddings(query)
    
    try:
        hits = qdrant_client.query_points(
            collection_name="poi_base_clip",
            query=v_math,
            limit=top_k
        ).points
        
        return {
            "query": query, 
            "results": [
                {
                    "path": h.payload.get("path"), 
                    "score": min(0.99, (h.score * 2.5) + 0.1),
                    "raw_score": h.score
                } for h in hits if h.score >= 0.22
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), model_type: str = "base_clip", top_k: int = 5):
    contents = await file.read()
    
    # Check if we should process image locally or cloud
    if local_model:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        v_math = local_model.encode(image, convert_to_numpy=True).tolist()
    else:
        v_math = get_embeddings(contents, is_image=True)
    
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
