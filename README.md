This project is a Person of Interest (POI) Image Search System designed to find specific people in a dataset (like CelebA) using natural language descriptions.

🚀 Project Overview
The system allows you to search through thousands of images by typing descriptions such as "a smiling woman with blonde hair and glasses" or "a man with a beard wearing a suit".

Core Components:
AI Engine (CLIP Model): It uses the clip-ViT-B-32 model. This is a powerful AI that understands both text and images. It converts images into mathematical "vectors" (embeddings) that represent their visual features.
Vector Search: When you type a query, it converts your text into a vector and uses Cosine Similarity to find the images that "mathematically" look most like your description.
Streamlit Frontend: A modern, premium web interface (

src/ad_poi.py
) where you can see a gallery of celebrities and perform real-time searches.
📂 Code Structure & File Functions

run_app.py
: The main entry point. Run this to launch the Streamlit application.

src/ad_poi.py
: The core logic. It handles:
Data Setup: Downloading images (currently set to HuggingFace's celebA).
Embedding Generation: Scanning images and saving their "AI fingerprints" into the 

data/
 folder.
Search UI: The interface with the search bar, gallery, and similarity scores.

data/
: Contains 

index_vectors.npy
 (the AI embeddings) and 

index_meta.parquet
 (the list of image paths).
docs/images/celebA/: The folder where the individual celebrity images are stored.
🛠️ How to Run with your CelebA Dataset
Since you mentioned you will be adding the CelebA dataset, here is the workflow you should follow:

Place your Images: Add your CelebA .jpg files into the docs/images/celebA/ directory.
Reset the Index: If you add new images, you must delete the existing files in the data/ folder (index_vectors.npy and index_meta.parquet). This forces the AI to "re-learn" and index your new images.
Run the Setup:
uv run python src/ad_poi.py
This will scan your new images and create the search index.
Launch the App:
python run_app.py
