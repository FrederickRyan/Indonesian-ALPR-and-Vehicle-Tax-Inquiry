# A Comparative Analysis of Deep Learning Pipelines for Real-Time Indonesian ALPR and Vehicle Tax Inquiry

**Authors:**
- Bryan Anthony – 2702283934
- Frederick Ryan Suryardi – 2702223033
- Nico Valerian Marcello – 2702250242

**Course:** COMP7116001 - Computer Vision

---

## 1. Introduction

The manual verification of vehicle tax status (*Pajak Kendaraan Bermotor*) is an inefficient process for both drivers and authorities. This project introduces an innovative solution to remind Banten drivers of their tax status by developing a high-performance Automatic License Plate Recognition (ALPR) system. The system is envisioned as a proof-of-concept for a device that could be deployed at key locations, such as toll road gates, to provide real-time tax status information as a public service.

The project's core is a comparative study of two deep learning pipelines to identify the optimal architecture for recognizing Indonesian license plates in real-time. The resulting system will enable a seamless "scan-and-check" workflow by integrating with automatable government data sources, thereby improving the efficiency and awareness of vehicle tax compliance.

### 1.1 Related Works

Significant research has been conducted on ALPR systems, including applications in Indonesia. Several studies have successfully employed YOLO architectures (e.g., YOLOv5, YOLOv8) for detecting Indonesian license plates, often achieving high precision and recall. While these works demonstrate the effectiveness of individual components, few perform a direct comparative analysis between different state-of-the-art detector backbones (CNN vs. Transformer) like YOLOv8 and RT-DETRv2 specifically for Indonesian plates. Furthermore, the integration of ALPR with a real-time, automated tax inquiry system using web scripting presents a novel application step beyond standard recognition tasks. This project differs by empirically comparing two distinct pipelines (YOLOv8+EasyOCR vs. RT-DETRv2+EasyOCR) and demonstrating a practical civic tech application through the tax status check integration.

---

## 2. Project Overview

This project compares two deep-learning pipelines for Indonesian license plate detection + recognition:

- **Pipeline A (CNN-based):** YOLOv8 (Ultralytics) for license-plate detection + EasyOCR/Custom OCR for text
- **Pipeline B (Transformer-based):** RT-DETRv2 (Hugging Face Transformers) for license-plate detection + EasyOCR/Custom OCR for text

The main deliverable is a **Streamlit app** you can run locally, with support for a **fine-tuned OCR model** optimized for Indonesian license plates.

---

## 3. Dataset Source and Description

The project utilizes the **Indonesian Plate Number from Multi-Sources dataset**, publicly available on Kaggle under the Apache 2.0 license. This dataset comprises diverse real-world images of Indonesian license plates captured under various lighting conditions, angles, and environmental settings, making it suitable for training robust detection and recognition models.

---

## 4. What's in this Repository

### Core Application Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI (main demo application) — run with `streamlit run app.py` |
| `inference.py` | Model loading, detection inference, and OCR logic |
| `custom_plate_ocr.py` | DeepSeek-OCR based Indonesian Plate OCR with LoRA fine-tuning |
| `requirements.txt` | Python dependencies |

### Model Weights & Checkpoints

| Path | Description |
|------|-------------|
| `best.pt` | YOLOv8 trained weights |
| `rtdetr_checkpoint/` | RT-DETRv2 fine-tuned checkpoint |
| `indonesian_plate_ocr_lora/` | LoRA adapter for fine-tuned DeepSeek-OCR |

### Training Notebooks

| File | Description |
|------|-------------|
| `Group_6_CV_Detection_Only.ipynb` | Detection model training (YOLOv8 & RT-DETRv2) |
| `Group_6_CV_DeepseekOCR.ipynb` | DeepSeek-OCR fine-tuning with LoRA |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | This file |
| `GPU_SETUP_FIX.md` | Troubleshooting guide for GPU/CUDA setup |

---

## 5. Requirements

- Windows 10/11
- Python 3.10+ (3.11 recommended; 3.12 should work too)

Optional:
- NVIDIA GPU + CUDA for faster inference (the project also runs on CPU)

---

## 6. Setup (Fresh Machine)

Open this folder in **VS Code**.

### A. Create a Virtual Environment

In VS Code Terminal (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### B. Install PyTorch with GPU Support (Important!)

**⚠️ Do NOT skip this step!** The default `pip install torch` installs CPU-only PyTorch.

For NVIDIA GPUs, install PyTorch with CUDA first:

```powershell
# For CUDA 12.1 (recommended for most modern NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Other CUDA versions:
```powershell
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### C. Install Other Dependencies

```powershell
pip install -r requirements.txt
```

---

## 7. Run the Streamlit Demo App

From the project folder (with `.venv` activated):

```powershell
streamlit run app.py
```

A browser tab will open. In the sidebar you can configure:

### Model Paths
- **YOLOv8 Weights Path** (default: `runs/detect/yolo_alpr_final/weights/best.pt`)
- **RT-DETRv2 Checkpoint Path** (default: `rtdetr_v2_results/checkpoint-2064`)
- **Custom OCR Model Path** (default: `indonesian_plate_ocr_model`)

### OCR Configuration
- The app will automatically use the **fine-tuned Indonesian Plate OCR** if the model folder exists
- Falls back to **EasyOCR** if custom model is not available

### Detection Thresholds
- **Detection Confidence Threshold** — General threshold for YOLO (default: 0.25)
- **RT-DETRv2 Specific Threshold** — Separate slider for RT-DETRv2 (default: 0.10)
  - *Note: RT-DETRv2 outputs lower confidence scores due to limited training, so it needs a lower threshold*

### Usage
1. Upload an image
2. Click detection buttons (YOLOv8 or RT-DETRv2)
3. The app crops detected plate region(s) and runs OCR
4. Optionally query Banten Province vehicle tax database

---

## 8. Notebook (.ipynb) — Training Pipeline

The notebooks are designed to run on **Google Colab with T4 GPU**.

### Training Notebooks

| Notebook | Description |
|----------|-------------|
| `Group_6_CV_Detection_Only.ipynb` | YOLOv8 and RT-DETRv2 detection training |
| `Group_6_CV_DeepseekOCR.ipynb` | DeepSeek-OCR fine-tuning with LoRA adapter |

### Training Times (T4 GPU)
- **YOLOv8:** ~5-8 minutes (10 epochs)
- **RT-DETRv2:** ~10-12 minutes (8 epochs)
- **DeepSeek-OCR LoRA:** ~15-20 minutes

### After Training
1. Download model weights from Colab
2. Place in project folder:
   - `best.pt` — YOLOv8 weights
   - `rtdetr_checkpoint/` — RT-DETRv2 checkpoint
   - `indonesian_plate_ocr_lora/` — LoRA adapter

---

## 9. Architecture Overview

### Detection Models

| Model | Type | Input Size | Training |
|-------|------|------------|----------|
| YOLOv8n | CNN (Ultralytics) | 640×640 | 10 epochs on Indonesian plates |
| RT-DETRv2 | Transformer (HuggingFace) | 640×640 | 8 epochs on Indonesian plates |

### OCR Models

| Model | Architecture | Description |
|-------|--------------|-------------|
| EasyOCR | CRNN-based | Default fallback, general-purpose |
| DeepSeek-OCR + LoRA | Vision-Language Model | Fine-tuned on Indonesian plate images |

### DeepSeek-OCR Architecture
The fine-tuned OCR model (`custom_plate_ocr.py`) uses:
- **Base Model:** `unsloth/DeepSeek-OCR` (Vision-Language Model)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) for efficient training
- **Quantization:** 4-bit quantization with bitsandbytes for low VRAM GPUs

---

## 10. Data Flow (High Level)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Image   │ ──▶ │  Detection Model │ ──▶ │  Cropped Plate  │
│                 │     │  (YOLO/RT-DETR)  │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Tax Inquiry    │ ◀── │    OCR Model     │ ◀── │   Preprocessed  │
│  (Banten Gov)   │     │ (Custom/EasyOCR) │     │     Plate       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 11. One-Command Run (After Setup)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```