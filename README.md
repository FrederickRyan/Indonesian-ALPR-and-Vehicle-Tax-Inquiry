Model & Checkpoint Download (Required)

The following folders are **excluded from GitHub** and must be downloaded manually:

- `rtdetr_checkpoint/` https://drive.google.com/drive/folders/1I8zPx4UWPfbDwUtxhPw3BdaXKpRMqyzJ?usp=drive_link
- `indonesian_plate_ocr_lora/` https://drive.google.com/drive/folders/1D9Vl3U1tQdESi6XDFDrsc4hm0Qhqo2wA?usp=sharing
- Model weight files (`*.pt`, `*.pth`)


After downloading, place the folders inside the project root directory:

```text
Computer Vision - LB02 - Group 6/
├── rtdetr_checkpoint/
├── indonesian_plate_ocr_lora/
```
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
| `app.py` | Streamlit UI (main demo application) |
| `inference.py` | Model loading, detection inference, and OCR logic |
| `custom_plate_ocr.py` | Custom Indonesian Plate OCR module (TPS-ResNet-BiLSTM-Attn architecture) |
| `requirements.txt` | Python dependencies |

### Model Weights & Checkpoints

| Path | Description |
|------|-------------|
| `runs/detect/yolo_alpr_final/weights/best.pt` | YOLOv8 trained weights |
| `rtdetr_v2_results/checkpoint-2064/` | RT-DETRv2 fine-tuned checkpoint |
| `indonesian_plate_ocr_model/` | Custom OCR model folder (after training in Colab) |

### Training & Experiments

| File | Description |
|------|-------------|
| `NEW_Group_6_Computer_Vision_Final_Project.ipynb` | Main Colab notebook for training all models |

### Utility Scripts

| File | Description |
|------|-------------|
| `debug_rtdetr_load.py` | Utility to verify RT-DETRv2 checkpoint loads correctly |
| `check_deps.py` | Streamlit dependency check page |
| `check_rtdetr.py` | Quick Transformers RT-DETR import verification |

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

### B. Install Packages

```powershell
pip install -r requirements.txt
```

If `torch` fails to install via `requirements.txt` on your machine, install it first using the official PyTorch selector, then re-run:

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

## 8. Optional: Quick Checks & Troubleshooting Tools

### A. Dependency Check Page

```powershell
streamlit run check_deps.py
```

This will display which libraries are installed and also writes a log file: `deps_status.txt`.

### B. Verify RT-DETR Imports

```powershell
python check_rtdetr.py
```

### C. Verify RT-DETR Checkpoint Loads

```powershell
python debug_rtdetr_load.py
```

You can also point it to another checkpoint folder:

```powershell
python debug_rtdetr_load.py --path "PATH\TO\checkpoint-folder"
```

---

## 9. Notebook (.ipynb) — Training Pipeline

The notebook `NEW_Group_6_Computer_Vision_Final_Project.ipynb` is designed to run on **Google Colab with T4 GPU**.

### Notebook Contents

| Cell # | Description |
|--------|-------------|
| 1-3 | Environment setup, Kaggle download, data preparation |
| 4-5 | **Pipeline A:** YOLOv8 training (10 epochs) |
| 6 | EasyOCR initialization |
| 7-8 | **Pipeline B:** RT-DETRv2 training (8 epochs, ~12 mins on T4) |
| 9 | Pipeline comparison summary |
| 10-11 | OCR fine-tuning setup (dependencies, clone repo, download pre-trained model) |
| 12 | Load and analyze Plate Text Dataset (1,863 samples) |
| 13-14 | Prepare LMDB format for OCR training |
| 15 | **Fine-tune OCR model** (TPS-ResNet-BiLSTM-Attn, 2000 iterations, ~10 mins) |
| 16 | Test fine-tuned OCR on validation images |
| 17 | **Export OCR model** for Streamlit deployment |
| 18+ | Vehicle Tax Inquiry module & full pipeline comparison |

### Training Times (T4 GPU)
- **YOLOv8:** ~5-8 minutes (10 epochs)
- **RT-DETRv2:** ~10-12 minutes (8 epochs)
- **OCR Fine-tuning:** ~10-15 minutes (2000 iterations)
- **Total:** ~25-35 minutes

### After Training
1. Download `indonesian_plate_ocr_model.zip` from Colab (or find in Google Drive if mounted)
2. Extract to project folder: `indonesian_plate_ocr_model/`
3. The Streamlit app will auto-detect and use the fine-tuned OCR

---

## 10. Architecture Overview

### Detection Models

| Model | Type | Input Size | Training |
|-------|------|------------|----------|
| YOLOv8n | CNN (Ultralytics) | 640×640 | 10 epochs on Indonesian plates |
| RT-DETRv2 | Transformer (HuggingFace) | 640×640 | 8 epochs on Indonesian plates |

### OCR Models

| Model | Architecture | Description |
|-------|--------------|-------------|
| EasyOCR | CRNN-based | Default fallback, general-purpose |
| Custom OCR | TPS-ResNet-BiLSTM-Attn | Fine-tuned on 1,863 Indonesian plate images |

### Custom OCR Architecture
The fine-tuned OCR model (`custom_plate_ocr.py`) uses:
- **TPS (Thin Plate Spline):** Spatial transformer for text rectification
- **ResNet:** Feature extraction backbone
- **BiLSTM:** Bidirectional sequence modeling
- **Attention:** Character-level attention decoder

---

## 11. Data Flow (High Level)

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

## 12. Common Issues

### “ModuleNotFoundError: No module named …”

You are likely using the wrong Python interpreter. Activate `.venv` and run again:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In VS Code you can also select interpreter:

- `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`

### TensorFlow oneDNN startup message

Some environments print a TensorFlow oneDNN notice even though we run PyTorch/Transformers.
It is safe to ignore.

If desired, you can disable it for your terminal session:

```powershell
$env:TF_ENABLE_ONEDNN_OPTS="0"
```

---

## 13. One-Command Run (After Setup)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
