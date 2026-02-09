from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from pathlib import Path
import requests
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

app = FastAPI(title="POI Search API - Cloud Optimized")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sentence-transformers/clip-ViT-B-32"

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
if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"📡 Connected to Qdrant Cloud")
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")

def get_hf_embeddings(inputs, is_image=False):
    """Offload AI math to Hugging Face to save Render RAM"""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN missing in environment variables")
    
    API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    if is_image:
        # Prepare image for API
        response = requests.post(API_URL, headers=headers, data=inputs)
    else:
        # Prepare text for API
        response = requests.post(API_URL, headers=headers, json={"inputs": [inputs]})
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HF API Error: {response.text}")
    
    # HF returns a nested list, we just want the vector
    res = response.json()
    if isinstance(res, list) and len(res) > 0:
        return res[0] if not is_image else res
    return res

@app.get("/")
async def root():
    return {
        "status": "Online", 
        "mode": "Cloud Inference (Ultra-Lite)", 
        "qdrant": qdrant_client is not None,
        "hf_api": HF_TOKEN is not None
    }

@app.post("/search/text")
async def search_text(query: str, model_type: str = "base_clip", top_k: int = 5):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant not connected")
    
    # Get embeddings from HF API (uses 0 RAM on Render)
    qv = get_hf_embeddings(query)
    
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), model_type: str = "base_clip", top_k: int = 5):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant not connected")
    
    contents = await file.read()
    
    # Get embeddings from HF API
    qv = get_hf_embeddings(contents, is_image=True)
    
    try:
        hits = qdrant_client.search(
            collection_name="poi_base_clip",
            query_vector=qv,
            limit=top_k
        )
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
