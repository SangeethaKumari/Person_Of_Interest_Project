from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoModel, AutoProcessor
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

app = FastAPI(title="POI Search API")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Mount static files to serve images
PROJECT_ROOT = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_CONFIG = {
    "base_clip": "clip-ViT-B-32",
    "enhanced_clip_l": "clip-ViT-L-14",
    "siglip2": "google/siglip2-base-patch16-224"
}

class Siglip2Encoder:
    def __init__(self, model_id):
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, data, normalize_embeddings=True):
        is_text = isinstance(data, str)
        if is_text:
            inputs = self.processor(text=data, padding="max_length", return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
        else:
            inputs = self.processor(images=data, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
        
        if normalize_embeddings:
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()

# Initialize Qdrant client if URL is provided
qdrant_client = None
if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"📡 Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")

# Global state for models and data
models = {}
meta_data = None
local_embeddings = {}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
META_PATH = DATA_DIR / "index_meta.parquet"

@app.on_event("startup")
async def startup_event():
    global meta_data
    print("🚀 Starting POI API...")
    
    # Load metadata
    if META_PATH.exists():
        meta_data = pd.read_parquet(META_PATH)
    else:
        print("⚠️ Warning: Metadata not found at index_meta.parquet")

    for key, model_id in MODELS_CONFIG.items():
        print(f"📥 Loading {key}...")
        try:
            if "siglip" in model_id.lower():
                models[key] = Siglip2Encoder(model_id)
            else:
                models[key] = SentenceTransformer(model_id)
            
            # Load local embeddings as fallback
            vec_path = DATA_DIR / f"index_vectors_{key.lower().replace('_', '-') if 'clip_l' in key else key}.npy"
            # Adjustment for naming inconsistency in my script
            if key == "enhanced_clip_l":
                vec_path = DATA_DIR / "index_vectors_enhanced_clip-l.npy"
            elif key == "siglip2":
                vec_path = DATA_DIR / "index_vectors_siglip_2.npy"
            elif key == "base_clip":
                vec_path = DATA_DIR / "index_vectors_base_clip.npy"

            if vec_path.exists():
                local_embeddings[key] = np.load(vec_path)
                print(f"✅ Loaded local embeddings for {key}")
        except Exception as e:
            print(f"❌ Error loading {key}: {e}")

@app.get("/")
async def root():
    return {"message": "POI Search API is running", "models": list(MODELS_CONFIG.keys()), "qdrant": qdrant_client is not None}

@app.post("/search/text")
async def search_text(query: str, model_type: str = "enhanced_clip_l", top_k: int = 5):
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    model = models[model_type]
    qv = model.encode(query).reshape(1, -1).flatten().tolist()
    
    if qdrant_client:
        try:
            hits = qdrant_client.search(
                collection_name=f"poi_{model_type}",
                query_vector=qv,
                limit=top_k
            )
            return {
                "query": query, 
                "model": model_type, 
                "results": [
                    {
                        "path": h.payload.get("path"), 
                        "score": min(0.99, (h.score * 2.5) + 0.1) if "clip" in model_type else h.score,
                        "raw_score": h.score
                    } for h in hits
                ]
            }
        except Exception as e:
            print(f"⚠️ Qdrant search failed: {e}, falling back to local")

    vecs = local_embeddings.get(model_type)
    if vecs is None:
        raise HTTPException(status_code=500, detail="Embeddings not available locally or on Qdrant")

    qv_np = np.array(qv).reshape(1, -1)
    sims = cosine_similarity(qv_np, vecs).ravel()
    top_indices = sims.argsort()[::-1][:top_k]
    
    return {
        "query": query, "model": model_type, 
        "results": [
            {
                "path": meta_data.loc[idx, "path"] if meta_data is not None else f"index_{idx}",
                "score": min(0.99, (float(sims[idx]) * 2.5) + 0.1) if "clip" in model_type else float(sims[idx]),
                "raw_score": float(sims[idx])
            } for idx in top_indices
        ]
    }

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), model_type: str = "enhanced_clip_l", top_k: int = 5):
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    model = models[model_type]
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    qv = model.encode(image).reshape(1, -1).flatten().tolist()
    
    if qdrant_client:
        try:
            hits = qdrant_client.search(
                collection_name=f"poi_{model_type}",
                query_vector=qv,
                limit=top_k
            )
            return {
                "model": model_type, 
                "results": [
                    {
                        "path": h.payload.get("path"), 
                        "score": min(0.99, (h.score * 1.5)),
                        "raw_score": h.score
                    } for h in hits
                ]
            }
        except Exception as e:
            print(f"⚠️ Qdrant search failed: {e}, falling back to local")

    vecs = local_embeddings.get(model_type)
    if vecs is None:
        raise HTTPException(status_code=500, detail="Embeddings not available locally or on Qdrant")

    qv_np = np.array(qv).reshape(1, -1)
    sims = cosine_similarity(qv_np, vecs).ravel()
    top_indices = sims.argsort()[::-1][:top_k]
    
    return {
        "model": model_type, 
        "results": [
            {
                "path": meta_data.loc[idx, "path"] if meta_data is not None else f"index_{idx}",
                "score": min(0.99, (float(sims[idx]) * 1.5)),
                "raw_score": float(sims[idx])
            } for idx in top_indices
        ]
    }
