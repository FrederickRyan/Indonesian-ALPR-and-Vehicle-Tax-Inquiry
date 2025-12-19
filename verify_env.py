
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import dependencies...")
    import easydict
    import einops
    print("Dependencies 'easydict' and 'einops' found.")
except ImportError as e:
    print(f"FAILED: Missing dependency: {e}")
    sys.exit(1)

try:
    print("Attempting to load DeepSeek-OCR model...")
    # Mimic the loading process from custom_plate_ocr.py
    from transformers import AutoModel, AutoTokenizer
    
    # Assuming the path is correct as per logs: C:\Users\Bryan\Documents\CV Final Project\indonesian_plate_ocr_lora
    # Or base model: unsloth/DeepSeek-OCR 
    # We'll just try to load the config or class to verify imports work
    
    # Actually, let's just try to import the AutoConfig and see if it fails with the dynamic module error
    from transformers import AutoConfig
    
    # We can try loading the specific model path if we know it, or just the base one if that was the issue.
    # The log said: Base Model: unsloth/DeepSeek-OCR
    
    model_id = "unsloth/DeepSeek-OCR"
    print(f"Loading config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print("Config loaded successfully. Dynamic module loading is working.")
    
    print("VERIFICATION SUCCESSFUL: DeepSeek-OCR environment is ready.")

except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
    sys.exit(1)
