
import os

# If TensorFlow is installed in the environment, it may print a oneDNN message
# even though this script uses PyTorch/Transformers. Setting this disables those
# TF oneDNN custom ops and silences that startup notice.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import torch
import traceback
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor


def _default_checkpoint_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "rtdetr_v2_results", "checkpoint-2064")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug loading an RT-DETRv2 checkpoint exported in this repo.",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("RTDETR_CHECKPOINT", _default_checkpoint_path()),
        help=(
            "Path to checkpoint folder (can also set RTDETR_CHECKPOINT env var). "
            "Defaults to rtdetr_v2_results/checkpoint-2064 relative to this file."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: cpu, cuda, cuda:0, etc. Defaults to auto.",
    )
    return parser.parse_args()


args = _parse_args()
path = os.path.abspath(args.path)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading from: {path}")
print(f"Device: {device}")

if not os.path.isdir(path):
    raise FileNotFoundError(
        f"Checkpoint directory not found: {path}\n"
        "Tip: pass --path <folder> or set RTDETR_CHECKPOINT env var."
    )

try:
    processor = RTDetrImageProcessor.from_pretrained(path)
    print("Processor loaded")
    
    model = RTDetrV2ForObjectDetection.from_pretrained(
        path, 
        num_labels=1, 
        ignore_mismatched_sizes=True
    ).to(device)
    print("Model loaded")
    
except Exception as e:
    print("FAILED")
    traceback.print_exc()
