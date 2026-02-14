import os
import requests
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
# Testing the router URL structure I saw in the client error
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {hf_token}", "X-Wait-For-Model": "true"}

payload = {
    "inputs": "a beautiful landscape",
}

print(f"Calling Router API: {API_URL}")
response = requests.post(API_URL, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")
