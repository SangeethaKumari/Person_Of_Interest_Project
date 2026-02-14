import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {hf_token}", "X-Wait-For-Model": "true"}

# Create a small dummy image if needed, but we have the real one
img_path = "/Users/sangeetha/Supportvector2026/llm/projects/Person_Of_Interest_Project/docs/images/celebA/train_000759.jpg"

with open(img_path, "rb") as f:
    img_data = f.read()

payload = {
    "inputs": "photorealistic professional portrait, refine this",
    "image": base64.b64encode(img_data).decode("utf-8"),
    "parameters": {"strength": 0.5, "guidance_scale": 7.5}
}

print(f"Calling HF API: {API_URL}")
response = requests.post(API_URL, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")
