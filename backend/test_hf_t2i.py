import os
import requests
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
# Try the standard URL again but with a T2I task which is most common
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {hf_token}", "X-Wait-For-Model": "true"}

payload = {
    "inputs": "a beautiful landscape painting",
}

print(f"Calling HF API (T2I): {API_URL}")
response = requests.post(API_URL, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")
