import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
META_PATH = DATA_DIR / "index_meta.parquet"

MODELS = {
    "base_clip": "index_vectors_base_clip.npy",
    "enhanced_clip_l": "index_vectors_enhanced_clip-l.npy",
    "siglip2": "index_vectors_siglip_2.npy"
}

def migrate():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    if not META_PATH.exists():
        print(f"❌ Metadata not found at {META_PATH}")
        return

    meta = pd.read_parquet(META_PATH)
    
    for model_key, vec_file in MODELS.items():
        vec_path = DATA_DIR / vec_file
        if not vec_path.exists():
            print(f"⚠️  Skipping {model_key}: {vec_path} not found")
            continue
            
        print(f"🧠 Indexing {model_key} to Qdrant...")
        vectors = np.load(vec_path)
        
        # Create collection
        vector_size = vectors.shape[1]
        collection_name = f"poi_{model_key}"
        
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        
        # Upload in batches
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc=f"Uploading {model_key}"):
            batch_vecs = vectors[i:i+batch_size].tolist()
            batch_meta = meta.iloc[i:i+batch_size].to_dict('records')
            
            points = [
                models.PointStruct(
                    id=i + j,
                    vector=v,
                    payload=batch_meta[j]
                )
                for j, v in enumerate(batch_vecs)
            ]
            
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
    print("✅ Migration complete!")

if __name__ == "__main__":
    migrate()
