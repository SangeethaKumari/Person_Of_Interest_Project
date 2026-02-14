from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import io
import os
import torch
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from better_profanity import profanity
from diffusers import AutoPipelineForImage2Image
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Optimization: Use CPU for local and limit threads
torch.set_num_threads(1)
profanity.load_censor_words()

class Siglip2Encoder:
    """Custom wrapper for SigLIP 2 since sentence-transformers has config bugs with it."""
    def __init__(self, model_id):
        from transformers import AutoModel, AutoProcessor
        import torch
        print(f"📥 Loading SigLIP 2 ({model_id})...")
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = "cpu" # Forcing CPU to avoid OOM
        self.model.to(self.device)
        self.model.eval()

    def encode(self, data, convert_to_numpy=True, normalize_embeddings=True):
        import torch
        # Handle both text and images
        is_text = isinstance(data, str) or (isinstance(data, list) and len(data) > 0 and isinstance(data[0], str))
        
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
            
        if convert_to_numpy:
            return features.cpu().numpy()
        return features

app = FastAPI(title="POI Search API - Triple Model Mode")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Model Definitions
MODELS_CONFIG = {
    "base_clip": {
        "id": "clip-ViT-B-32",
        "collection": "poi_base_clip",
        "name": "Base CLIP (ViT-B-32)",
        "threshold": 0.22,
        "multiplier": 2.5
    },
    "enhanced_clip_l": {
        "id": "clip-ViT-L-14",
        "collection": "poi_enhanced_clip_l",
        "name": "Enhanced CLIP-L (ViT-L-14)",
        "threshold": 0.20,
        "multiplier": 1.9
    },
    "siglip2": {
        "id": "google/siglip2-base-patch16-224",
        "collection": "poi_siglip2",
        "name": "SigLIP 2 (Google)",
        "threshold": 0.01, # SigLIP 2 raw scores are extremely low (0.05 range)
        "multiplier": 12.0 # Significant boost to make progress bars visible
    }
}

# Global clients/models
qdrant_client = None
models = {}
refinement_pipe = None

@app.on_event("startup")
async def startup_event():
    global qdrant_client, models
    
    # 1. Connect to Qdrant
    if QDRANT_URL:
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print("📡 Connected to Qdrant Cloud")
        except Exception as e:
            print(f"⚠️ Qdrant Error: {e}")
    else:
        print("⚠️ QDRANT_URL not found in environment variables. Database not connected.")

    # 2. Load Models Locally
    for key, config in MODELS_CONFIG.items():
        try:
            if key == "siglip2":
                models[key] = Siglip2Encoder(config['id'])
            else:
                print(f"📥 Loading {config['name']} ({config['id']})...")
                models[key] = SentenceTransformer(config['id'], device="cpu")
            print(f"✅ {config['name']} Loaded")
        except Exception as e:
            print(f"❌ Error loading {key}: {e}")

    # 3. Load Local Refinement Model (SDXL Turbo)
    try:
        global refinement_pipe
        print("📥 Loading SDXL Turbo for local refinement...")
        refinement_pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        refinement_pipe.to("mps")
        print("✅ SDXL Turbo Loaded on MPS")
    except Exception as e:
        print(f"⚠️ SDXL Turbo Load Error: {e} (Refinement will be disabled)")

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

async def perform_search(inputs, is_image=False, top_k=5):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Database not connected")
        
    all_results = {}
    
    for key, model_obj in models.items():
        config = MODELS_CONFIG[key]
        
        try:
            # Encode
            qv = model_obj.encode(inputs, convert_to_numpy=True)
            # Ensure it's a list for Qdrant
            if hasattr(qv, "tolist"):
                qv = qv.tolist()
            if isinstance(qv, list) and len(qv) > 0 and isinstance(qv[0], list):
                qv = qv[0]
            
            hits = qdrant_client.query_points(
                collection_name=config['collection'],
                query=qv,
                limit=top_k
            ).points
            
            threshold = config['threshold']
            multiplier = config['multiplier']
            
            results = []
            for h in hits:
                if h.score >= threshold:
                    visual_score = (h.score * multiplier) + 0.05 if not is_image else (h.score * 1.5)
                    results.append({
                        "path": h.payload.get("path"), 
                        "score": min(0.99, visual_score),
                        "raw_score": h.score
                    })
            
            all_results[key] = results
            print(f"🔍 {config['name']} search for '{inputs if isinstance(inputs, str) else 'image'}': found {len(results)} matches (Threshold: {threshold})")
            
        except Exception as e:
            print(f"❌ Search error for {key}: {e}")
            all_results[key] = []
            
    return all_results

@app.get("/")
async def root():
    return {
        "status": "Online", 
        "models_active": list(models.keys()),
        "info": "Triple Model Evaluation Server"
    }

@app.post("/search/all")
async def search_all(query: str, top_k: int = 5):
    if profanity.contains_profanity(query):
        raise HTTPException(status_code=400, detail="Inappropriate language detected")
    return await perform_search(query, is_image=False, top_k=top_k)

@app.post("/search/text")
async def search_text(query: str, model_type: str = "base_clip", top_k: int = 5):
    res = await perform_search(query, is_image=False, top_k=top_k)
    if model_type not in res:
        return {"query": query, "results": res.get("base_clip", [])}
    return {"query": query, "results": res.get(model_type, [])}

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), model_type: str = "base_clip", top_k: int = 5):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    res = await perform_search(image, is_image=True, top_k=top_k)
    
    if model_type == "all":
        return res
@app.post("/search/refine")
async def refine_image(image_path: str, prompt: str):
    """
    Step 2: Refinement. Takes a retrieved image and uses Local SDXL Turbo
    to regenerate the face based on feedback.
    """
    import base64
    from io import BytesIO
    
    if refinement_pipe is None:
        raise HTTPException(status_code=503, detail="Generative Engine not loaded locally")

    try:
        # 1. Load the original image from local disk
        full_path = PROJECT_ROOT / image_path
        if not full_path.exists():
             raise HTTPException(status_code=404, detail="Original image not found")
             
        init_image = Image.open(full_path).convert("RGB").resize((512, 512))

        # 2. Local Generative Inference
        # SDXL Turbo is fast (1-4 steps) and needs no guidance_scale
        generated_image = refinement_pipe(
            prompt, 
            image=init_image, 
            strength=0.5, 
            guidance_scale=0.0, 
            num_inference_steps=2
        ).images[0]
        
        # 3. Convert to Base64
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "refined_image": img_str,
            "status": "success",
            "message": f"POI Refined Locally: {prompt}"
        }
    except Exception as e:
        print(f"Error in Refinement: {e}")
        raise HTTPException(status_code=500, detail=str(e))
