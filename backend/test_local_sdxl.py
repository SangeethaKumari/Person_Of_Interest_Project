import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import os

def test_local_refine():
    # 1. Setup Model (Using SDXL Turbo for speed on Mac)
    model_id = "stabilityai/sdxl-turbo"
    
    print(f"📥 Loading {model_id} to Mac GPU (MPS)...")
    # Use float16 and move to 'mps' (Apple Silicon GPU)
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    pipe.to("mps") 

    # 2. Load a sample image from your dataset
    init_image_path = "docs/images/celebA/train_001055.jpg"
    if not os.path.exists(init_image_path):
        print("❌ Image not found. Please check the path.")
        return

    init_image = load_image(init_image_path).resize((512, 512))

    # 3. Refine the image
    prompt = "A cinematic portrait of a person with a very angular face and high cheekbones, professional lighting"
    
    print("🎨 Generating refined POI locally...")
    # SDXL Turbo only needs 1-4 steps!
    image = pipe(
        prompt, 
        image=init_image, 
        strength=0.5, 
        guidance_scale=0.0, 
        num_inference_steps=2
    ).images[0]

    # 4. Save result
    image.save("local_refinement_test.png")
    print("✅ Success! Check 'local_refinement_test.png' on your desktop.")

if __name__ == "__main__":
    test_local_refine()
