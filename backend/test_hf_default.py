import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

img_path = "/Users/sangeetha/Supportvector2026/llm/projects/Person_Of_Interest_Project/docs/images/celebA/train_000759.jpg"

with open(img_path, "rb") as f:
    img_data = f.read()

print("Calling HF InferenceClient (default model for image-to-image)")
try:
    image = client.image_to_image(
        img_data,
        prompt="photorealistic professional portrait, refine this",
        model="runwayml/stable-diffusion-v1-5",
        strength=0.5
    )
    print("Success!")
    image.save("refined_test_default.png")
except Exception as e:
    import traceback
    traceback.print_exc()
