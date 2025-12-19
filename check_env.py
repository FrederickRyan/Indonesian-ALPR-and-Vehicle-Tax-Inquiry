
import sys
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("-" * 30)
print("Attempting to import torch...")
try:
    import torch
    print(f"✅ torch {torch.__version__} installed (CUDA available: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"❌ torch failed: {e}")
except Exception as e:
    print(f"❌ torch error: {e}")

print("-" * 30)
print("Attempting to import transformers...")
try:
    import transformers
    print(f"✅ transformers {transformers.__version__} installed")
except ImportError as e:
    print(f"❌ transformers failed: {e}")
except Exception as e:
    print(f"❌ transformers error: {e}")

print("-" * 30)
print("Attempting to import easyocr...")
try:
    import easyocr
    print(f"✅ easyocr installed")
except ImportError as e:
    print(f"❌ easyocr failed: {e}")
except Exception as e:
    print(f"❌ easyocr error: {e}")

print("-" * 30)
print("Attempting to import peft...")
try:
    import peft
    print(f"✅ peft {peft.__version__} installed")
except ImportError as e:
    print(f"❌ peft failed: {e}")

print("-" * 30)
print("Checking streamlit...")
try:
    import streamlit
    print(f"✅ streamlit {streamlit.__version__} installed")
except ImportError as e:
    print(f"❌ streamlit failed: {e}")
