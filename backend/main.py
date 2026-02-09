from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from PIL import Image
import io
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from better_profanity import profanity
from sentence_transformers import SentenceTransformer

# Safety filters
profanity.load_censor_words()

app = FastAPI(title="POI Search API - Nano Mode")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MODEL_ID = "clip-ViT-B-32"

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
            print(f"❌ Qdrant Error: {e}")

    # 2. Load Model (Optimized for 512MB RAM)
    try:
        print(f"📥 Loading Nano {MODEL_ID}...")
        # device='cpu' is mandatory for Render Free tier
        model = SentenceTransformer(MODEL_ID, device="cpu")
        print("✅ Nano Model Ready")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")

@app.get("/")
async def root():
    return {"status": "Online", "mode": "Nano Local", "ready": model is not None}

@app.post("/search/text")
async def search_text(query: str, model_type: str = "base_clip", top_k: int = 5):
    if not model:
        raise HTTPException(status_code=500, detail="Model still loading...")
    
    if profanity.contains_profanity(query):
        raise HTTPException(status_code=400, detail="Inappropriate language")

    # Encode locally (Zero reliability issues, zero 404s)
    qv = model.encode(query, convert_to_numpy=True).tolist()
    
    try:
        hits = qdrant_client.query_points(
            collection_name="poi_base_clip",
            query=qv,
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
    if not model:
        raise HTTPException(status_code=500, detail="Model still loading...")

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
        raise HTTPException(status_code=500, detail=str(e))
