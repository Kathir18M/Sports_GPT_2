import io
import os
import sys
import requests
from PIL import Image

# Ensure backend root is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from litellm import image_generation
from app.core.config import settings

async def generate_image(prompt: str) -> Image.Image:
    """
    Generates an image based on the prompt using Gemini's image generation model,
    with a seamless fallback to Pollinations AI if Gemini free-tier quota is reached.
    """
    models_to_try = [
        "gemini/gemini-2.5-flash-image",
        "huggingface/black-forest-labs/FLUX.1-schnell",
        "huggingface/black-forest-labs/FLUX.1-dev"
    ]

    for model in models_to_try:
        try:
            response = image_generation(
                model=model,
                prompt=prompt,
                api_key=settings.GEMINI_API_KEY,
                n=1
            )
            image_url = response.data[0].url
            img_response = requests.get(image_url, timeout=15)
            img_response.raise_for_status()
            return Image.open(io.BytesIO(img_response.content))
        except Exception as e:
            print(f"[Warning] Gemini image model '{model}' failed ({e}). Trying next...")

    # Fallback to Pollinations AI if Gemini free-tier rate limit/quota is reached
    print("[Info] Falling back to Pollinations AI image generation...")
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    img_response = requests.get(pollinations_url, timeout=30)
    img_response.raise_for_status()
    return Image.open(io.BytesIO(img_response.content))

if __name__ == "__main__":
    import os
    import asyncio
    
    async def main():
        prompt = "Cristiono ronaldo With World Cup"
        print(f"Generating image for prompt: '{prompt}'...")
        try:
            image = await generate_image(prompt)
            
            # Ensure the outputs directory exists
            os.makedirs("outputs", exist_ok=True)
            output_path = "outputs/generated_image.png"
            
            # Save the PIL Image
            image.save(output_path)
            print(f"\nSuccess! You can find the generated image at:\n{os.path.abspath(output_path)}")
        except Exception as e:
            print(f"Error during image generation: {e}")
            
    asyncio.run(main())