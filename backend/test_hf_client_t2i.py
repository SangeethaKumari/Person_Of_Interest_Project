import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

print("Calling InferenceClient.text_to_image with default model")
try:
    image = client.text_to_image("a beautiful landscape")
    print("Success!")
    image.save("t2i_success.png")
except Exception as e:
    import traceback
    traceback.print_exc()
