import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MODEL_ID = "google/siglip2-base-patch16-224"
IMG_DIR = Path("docs/images/celebA")
META_PATH = Path("data/index_meta.parquet")

def fix_siglip():
    # 1. Load Model
    print(f"📥 Loading {MODEL_ID}...")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    device = "cpu" 
    model.to(device)
    model.eval()

    # 2. Get images
    meta = pd.read_parquet(META_PATH)
    image_paths = [Path(p) for p in meta['path']]
    
    # 3. Generate Vectors
    print(f"🧠 Generating NEW embeddings for {len(image_paths)} images...")
    new_vectors = []
    for p in tqdm(image_paths):
        full_path = Path(__file__).resolve().parent / p
        try:
            img = Image.open(full_path).convert("RGB")
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                v = model.get_image_features(**inputs)
                v = torch.nn.functional.normalize(v, p=2, dim=1).cpu().numpy().tolist()[0]
            new_vectors.append(v)
        except Exception as e:
            print(f"Error on {p}: {e}")
            new_vectors.append([0.0] * 768)

    # 4. Upload to Qdrant
    print("📡 Uploading to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = "poi_siglip2"
    
    # Recreate collection to be safe
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE)
    )
    
    batch_size = 100
    for i in range(0, len(new_vectors), batch_size):
        batch_vecs = new_vectors[i:i+batch_size]
        batch_meta = meta.iloc[i:i+batch_size].to_dict('records')
        
        points = [
            qmodels.PointStruct(
                id=i + j,
                vector=v,
                payload=batch_meta[j]
            )
            for j, v in enumerate(batch_vecs)
        ]
        
        client.upsert(collection_name=collection_name, points=points)
    
    print("✅ SigLIP 2 Index Successfully Refreshed!")

if __name__ == "__main__":
    fix_siglip()
