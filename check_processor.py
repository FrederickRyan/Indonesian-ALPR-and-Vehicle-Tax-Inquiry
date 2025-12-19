
import torch
from transformers import AutoProcessor, AutoConfig
from PIL import Image
import numpy as np

model_id = "unsloth/DeepSeek-OCR"

try:
    print(f"Attempting to load AutoProcessor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✓ AutoProcessor loaded successfully.")
    
    # Create dummy image
    image = Image.new('RGB', (128, 128), color='red')
    prompt = "<image>\nTest."
    
    print("Testing processor call...")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    print("Keys in processed inputs:", inputs.keys())
    if "pixel_values" in inputs:
        print("✓ 'pixel_values' present. Shape:", inputs["pixel_values"].shape)
    
    if "images" in inputs and torch.is_tensor(inputs["images"]):
         print("✓ 'images' key is a tensor (some models use this instead of pixel_values). Shape:", inputs["images"].shape)
         
except Exception as e:
    print(f"❌ Failed to load or use AutoProcessor: {e}")
    import traceback
    traceback.print_exc()

