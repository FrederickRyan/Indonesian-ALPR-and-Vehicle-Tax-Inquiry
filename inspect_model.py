
import sys
import os
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Loading model to inspect attributes...")
    base_model_id = "unsloth/DeepSeek-OCR"
    lora_path = "C:\\Users\\Bryan\\Documents\\CV Final Project\\indonesian_plate_ocr_lora"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # Load base model
    model = AutoModel.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Base Configuration class: {model.config.__class__.__name__}")
    print(f"Base Model class: {model.__class__.__name__}")
    
    # Check for 'infer' method on base model
    if hasattr(model, 'infer'):
        print("✓ Base model has 'infer' method.")
    else:
        print("❌ Base model DOES NOT have 'infer' method.")
        
    # Load LoRA
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Check for 'infer' method on PeftModel (it might not be forwarded automatically)
    if hasattr(model, 'infer'):
        print("✓ PeftModel has 'infer' method.")
    elif hasattr(model.base_model, 'infer'):
        print("✓ PeftModel.base_model has 'infer' method.")
    elif hasattr(model.base_model.model, 'infer'):
        print("✓ PeftModel.base_model.model has 'infer' method.")
    else:
        print("❌ Could not find 'infer' method on PeftModel or its base.")

    # Inspect `forward` signature if possible
    # import inspection
    # print(inspect.signature(model.forward))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
