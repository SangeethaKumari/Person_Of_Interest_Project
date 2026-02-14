# Person of Interest (POI) AI Search System 🌟

An enterprise-grade, multi-model AI search engine for human faces. This project uses state-of-the-art vision-language models (CLIP and SigLIP 2) to enable semantic search through image datasets.

## 🚀 Architecture for Deployment

The project has been upgraded to a modern **Client-Server Architecture**, designed for high scalability and cloud deployment:

- **Frontend (React)**: Hosted on **Vercel**. A premium dark-themed UI built with Vite, Framer Motion, and Tailwind CSS.
- **Backend (FastAPI)**: Hosted on **Render**. Provides high-performance AI inference endpoints.
- **Vector DB (Qdrant)**: Hosted on **Qdrant Cloud**. Handles lightning-fast vector similarity searches.
- **Storage**: Images can be served via CDN or hosted documentation.

---

## 🛠️ Deployment Guide

### 1. Vector Database (Qdrant)
1. Sign up for a free [Qdrant Cloud](https://cloud.qdrant.io/) account.
2. Create a new Cluster and get your **URL** and **API Key**.
3. Run the migration script locally to push your existing embeddings:
   ```bash
   export QDRANT_URL="your-cluster-url"
   export QDRANT_API_KEY="your-api-key"
   cd backend && uv run python migrate_to_qdrant.py
   ```

### 2. Backend (Render)
1. Connect your GitHub repository to [Render.com](https://render.com/).
2. Create a new **Web Service**.
3. **CRITICAL**: Because this app runs heavy AI models (CLIP-L, SigLIP), you **must** select a plan with at least **2GB-4GB of RAM** (Starter or Pro). The "Free" tier will crash.
4. Set the **Root Directory** to `backend`.
5. The build and start commands will be automatically detected if you use the provided `render.yaml` blueprint, or set them manually:
   - Build Command: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
   - Start Command: `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variables:
   - `QDRANT_URL`: Your Qdrant cluster URL.
   - `QDRANT_API_KEY`: Your Qdrant API key.

### 3. Frontend (Vercel)
1. Import your repository into [Vercel](https://vercel.com/).
2. Set the **Root Directory** to `frontend`.
3. Set the Framework Preset to **Vite**.
4. Add Environment Variables:
   - `VITE_API_URL`: The URL of your Render backend (e.g., `https://poi-backend.onrender.com`).

---

## 💻 Local Development

### Prerequisites
- Python 3.12+
- Node.js 18+
- `uv` (Fast Python package manager)

### Start Backend
```bash
cd backend
uv run uvicorn main:app --reload
```

### Start Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🎭 Key Features
- **Narrative Search**: Find people using natural language descriptions.
- **Celebrity Doppelgänger**: Upload a photo to find your celebrity twin.
- **Multi-Model Comparison**: Contrast results from CLIP-B, CLIP-L, and SigLIP 2.
- **Glassmorphism UI**: A premium, responsive design with smooth animations.

---

## 📂 Code Structure
- `/backend`: FastAPI server, AI model logic, and migration scripts.
- `/frontend`: React application (Vite + Tailwind).
- `/src`: Legacy Streamlit implementation and local processing scripts.
- `/data`: Local vector embeddings and metadata.
- `/docs/images/celebA`: Image dataset storage.

---

Base CLIP (Blue):

Model ID: clip-ViT-B-32
Role: A balanced, industry-standard model from OpenAI. It is optimized for speed and general keyword matching.
Enhanced CLIP-L (Purple):

Model ID: clip-ViT-L-14
Role: A significantly larger version of CLIP. It has 303 Million parameters (compared to 86M in the base model), allowing it to see much finer facial details and textures, which is why it often gives you the "correct" results.
SigLIP 2 (Emerald):

Model ID: google/siglip2-base-patch16-224
Role: Google's latest high-end "Vision-Language" model. I have manually refreshed your Cloud Database to use the official SigLIP 2 weights, ensuring the embeddings are now perfectly matched to the model's "brain."

Built with ⚡ by **SupportVectors**



A silhouette against a sunset

A person walking through a market

A celebrity with a very intense and focused gaze
an expression of pure, contagious joy


. Introduction
The "Hook": Start by asking, "How do we typically find images?" (Usually keywords/tags). Explain that this project moves beyond tags to Semantic Search—finding images based on meaning, description, and visual similarity.
What it is: A multi-model AI search engine designed specifically for identifying human faces and attributes in large datasets.
2. Core Features (The "What")
🧠 Narrative Search: Users can search using natural language (e.g., "A celebrity with a very intense and focused gaze") rather than rigid keywords.
🔍 Reverse Image Search: Upload a photo to find "doppelgängers" or similar potential matches in the database.
📊 Multi-Model Evaluation: A unique feature that runs three competitive AI models simultaneously (CLIP Base, CLIP Large, Google SigLIP 2) to compare which "sees" the query best.
✨ Generative Refinement (New!): The ability to take a search result and modify it using Generative AI (e.g., "Make them look older" or "Add angular features") using a locally running Stable Diffusion XL Turbo model.
3. Technical Architecture (The "How")
Frontend: Built with React + Vite. Uses Tailwind CSS for a modern "Glassmorphism" aesthetic (translucent, premium UI).
Backend: FastAPI (Python). Acts as the orchestration layer between the UI, the AI models, and the database.
Database: Qdrant (Vector Database). Stores high-dimensional "embeddings" (mathematical representations) of faces, allowing for lightning-fast similarity searches across thousands of images.
AI Engine:
Vision Encoders: Uses sentence-transformers to convert text/images into vectors.
Generative Engine: Uses diffusers + SDXL Turbo running locally on Mac MPS (Metal Performance Shaders) for near-instant image manipulation without external API costs.
4. The "Triple Model" Strategy
explain why you use three models:

Base CLIP (ViT-B-32): The industry standard. Fast, balanced, good for general purpose.
Enhanced CLIP-L (ViT-L-14): A larger, more powerful model (303M parameters). Sees finer textures and details.
Google SigLIP 2: The cutting edge. Google's latest vision-language model, offering state-of-the-art semantic understanding.
5. Live Demo Flow (Recommended)
Search: Type a complex query like "A person looking joyful in a crowd".
Compare: Show how the three columns (Blue, Purple, Emerald) return slightly different results based on their model architecture.
Refine: Click a result, hit "Refine," and type "Make the lighting more dramatic". Show the SDXL Turbo generation happening in seconds.
6. Challenges & Solutions
Challenge: Remote APIs for checking "Refinement" were slow or deprecated.
Solution: Migrated to Local Inference. Leveraged the Mac's GPU (MPS) to run Stable Diffusion locally, reducing latency and removing dependency on external services.
Challenge: Managing large model weights.
Solution: Optimized startup scripts to load models only once and keep them in memory.
7. Future Roadmap
Scale: Deploying to a GPU cloud cluster (like Render or AWS) for production handling.
Video: Expanding from static images to video frame search.
Face Recognition: Adding a specific "Identity Verification" layer on top of the semantic search.
Speaker Tips / "Soundbites"
"We aren't just matching pixels; we are matching concepts."
"By running three models in parallel, we don't just get one answer—we get a consensus."
"We brought the generative cloud down to the edge, running SDXL directly on the hardware for privacy and speed."
