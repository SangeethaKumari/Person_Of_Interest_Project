import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import base64

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

model_id = "timbrooks/instruct-pix2pix"
img_path = "/Users/sangeetha/Supportvector2026/llm/projects/Person_Of_Interest_Project/docs/images/celebA/train_000759.jpg"

with open(img_path, "rb") as f:
    img_data = f.read()

print(f"Calling HF InferenceClient for model: {model_id}")
try:
    # InferenceClient.image_to_image returns a PIL Image
    image = client.image_to_image(
        img_data,
        prompt="photorealistic professional portrait, refine this",
        model=model_id,
        strength=0.5,
        guidance_scale=7.5
    )
    print("Success!")
    image.save("refined_test.png")
    print("Saved refined_test.png")
except Exception as e:
    print(f"Error: {e}")
