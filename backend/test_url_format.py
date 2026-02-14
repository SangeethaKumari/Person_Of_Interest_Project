
import os
import requests
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {hf_token}", "X-Wait-For-Model": "true"}

def test_url(url):
    print(f"Testing URL: {url}")
    payload = {"inputs": "a test query"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS")
        else:
            print(f"FAILED: {response.text[:100]}")
    except Exception as e:
        print(f"ERROR: {e}")

# URL 1: Standard replacement
url1 = "https://router.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
test_url(url1)

# URL 2: With /hf-inference/ path
url2 = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
test_url(url2)
