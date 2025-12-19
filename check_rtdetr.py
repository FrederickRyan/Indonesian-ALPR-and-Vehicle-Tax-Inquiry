import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import transformers
print(f"Transformers version: {transformers.__version__}")

try:
    from transformers import RTDetrV2ForObjectDetection
    print("RTDetrV2ForObjectDetection: IMPORT SUCCESS")
except ImportError as e:
    print(f"RTDetrV2ForObjectDetection: IMPORT FAILED - {e}")

try:
    from transformers import RTDetrForObjectDetection
    print("RTDetrForObjectDetection: IMPORT SUCCESS")
except ImportError as e:
    print(f"RTDetrForObjectDetection: IMPORT FAILED - {e}")
