import streamlit as st

# Set page configuration (Must be first!)
st.set_page_config(page_title="ALPR System Group 6", layout="wide")

from PIL import Image
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import sys
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, List

# Add current directory to path so imports work reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    import inference
except ImportError as e:
    st.error(f"Error importing inference module: {e}")
    inference = None

# Title and Description
st.title("🇮🇩 Indonesian Automatic License Plate Recognition")
st.markdown("""
**Group 6** | Computer Vision Final Project
Comparing two deep learning pipelines: **YOLOv8** vs **RT-DETRv2** for license plate detection and recognition.
""")


def _normalize_plate_for_query(plate_string: str) -> Tuple[Optional[str], Optional[dict]]:
    """Normalize plate text and parse into (kode, nomor, seri) for Banten query."""
    if not plate_string:
        return None, {"Status": "Error", "Message": "Empty plate string."}

    clean_plate = plate_string.upper().replace(" ", "")
    clean_plate = "".join(ch for ch in clean_plate if ch.isalnum())

    match = re.match(r"([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})", clean_plate)
    if not match:
        return None, {"Status": "Error", "Message": "Invalid Plate Format (Regex Fail)"}

    kode, nomor, seri = match.groups()
    return clean_plate, {"kode": kode, "nomor": nomor, "seri": seri}


def check_banten_tax_live(plate_string: str) -> dict:
    """Live query to https://infopkb.bantenprov.go.id/ using the recognized plate string."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as e:
        return {
            "Status": "Error",
            "Message": "Missing dependency. Please install 'requests' and 'beautifulsoup4'.",
            "Detail": str(e),
        }

    normalized, parsed = _normalize_plate_for_query(plate_string)
    if normalized is None:
        return parsed

    kode, nomor, seri = parsed["kode"], parsed["nomor"], parsed["seri"]

    base_url = "https://infopkb.bantenprov.go.id"
    url = f"{base_url}/p_infopkb.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"{base_url}/index.php",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Origin": base_url,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    payload = {
        "kode": kode,
        "nomor": nomor,
        "seri": seri,
        "tgl": datetime.now().strftime("%d-%m-%Y"),
        "index": "index.php",
    }

    try:
        session = requests.Session()
        # First visit the homepage to get any cookies (mimics browser behavior)
        session.get(f"{base_url}/index.php", headers=headers, timeout=15, verify=True)
        # Now make the actual POST request
        response = session.post(url, data=payload, headers=headers, timeout=30, verify=True)

        if "DATA KENDARAAN TIDAK ADA" in response.text.upper():
            return {"Status": "Not Found", "Message": "Plate number not found in database."}

        soup = BeautifulSoup(response.text, "html.parser")
        result_data: Dict[str, str] = {}

        target_keys = [
            "NO. POLISI",
            "MEREK",
            "TIPE/MODEL",
            "TAHUN / CC / BBM",
            "TGL. AKHIR PKB",
            "JUMLAH",
            "WARNA",
        ]

        rows = soup.find_all("div", class_="row")
        for row in rows:
            # Use recursive=False to only get direct children, avoiding duplicate nested rows
            key_div = row.find("div", class_=lambda x: x and "col-4" in x, recursive=False)
            val_div = row.find("div", class_=lambda x: x and "col-8" in x, recursive=False)
            if not key_div or not val_div:
                continue

            raw_key = key_div.get_text(" ", strip=True).upper().replace(":", "")
            # Get only direct text content to avoid duplicates from nested elements
            raw_val = val_div.get_text(" ", strip=True)
            # If value is duplicated (e.g., "HONDA HONDA"), take only the first half
            words = raw_val.split()
            if len(words) >= 2 and len(words) % 2 == 0:
                half = len(words) // 2
                first_half = words[:half]
                second_half = words[half:]
                if first_half == second_half:
                    raw_val = " ".join(first_half)

            if any(t in raw_key for t in target_keys):
                clean_key = next(t for t in target_keys if t in raw_key)
                result_data[clean_key] = raw_val

        if not result_data:
            return {"Status": "Unknown", "Message": "Page loaded but no data parsed. Layout may have changed."}

        return {"Status": "Success", "Data": result_data, "Queried": f"{kode} {nomor} {seri}"}

    except requests.exceptions.RequestException as e:
        return {"Status": "Error", "Message": "Network error while querying Banten site.", "Detail": str(e)}
    except Exception as e:
        return {"Status": "Error", "Message": "Unexpected error while parsing response.", "Detail": str(e)}

# Sidebar for Model Configuration
st.sidebar.header("Model Configuration")

# Define default paths relative to this script
DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "best.pt")
DEFAULT_RTDETR_PATH = os.path.join(BASE_DIR, "rtdetr_checkpoint")
DEFAULT_CUSTOM_OCR_PATH = os.path.join(BASE_DIR, "indonesian_plate_ocr_lora")

yolo_path = st.sidebar.text_input("YOLOv8 Weights Path", DEFAULT_YOLO_PATH)
rtdetr_path = st.sidebar.text_input("RT-DETRv2 Checkpoint Path", DEFAULT_RTDETR_PATH)

# Custom OCR Model Settings
st.sidebar.subheader("OCR Configuration")

# Check if CUDA is available - if not, warn about slow custom OCR
try:
    import torch
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False

# Toggle for custom OCR - default OFF if no CUDA
use_custom_ocr = st.sidebar.checkbox(
    "Use Fine-tuned DeepSeek OCR",
    value=False,  # Default OFF - DeepSeek OCR is very slow on CPU
    help="⚠️ DeepSeek OCR is VERY slow on CPU (5+ mins). Enable only with CUDA/GPU."
)

if use_custom_ocr and not cuda_available:
    st.sidebar.warning("⚠️ No GPU detected! DeepSeek OCR will be VERY slow (5+ mins per image).")

custom_ocr_path = st.sidebar.text_input(
    "Custom OCR Model Path", 
    DEFAULT_CUSTOM_OCR_PATH if use_custom_ocr else "",
    help="Path to fine-tuned Indonesian Plate OCR model. Leave empty to use EasyOCR.",
    disabled=not use_custom_ocr
)

# Check existence
yolo_exists = os.path.exists(yolo_path)
rtdetr_exists = os.path.exists(rtdetr_path)
custom_ocr_exists = os.path.exists(custom_ocr_path) if custom_ocr_path else False

if not yolo_exists:
    st.sidebar.warning(f"⚠️ YOLO file not found at: `{yolo_path}`")
if not rtdetr_exists:
    st.sidebar.warning(f"⚠️ RT-DETR file not found at: `{rtdetr_path}`")
if use_custom_ocr:
    if custom_ocr_exists:
        st.sidebar.success("✓ Custom OCR model found (will be slow on CPU!)")
    else:
        st.sidebar.warning("⚠️ Custom OCR path not found")
else:
    st.sidebar.info("ℹ️ Using EasyOCR (fast)")

# Threshold Sliders
st.sidebar.subheader("Detection Thresholds")
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Adjust this to filter out weak detections. RT-DETRv2 may need lower values (0.10-0.20)."
)

# Separate threshold for RT-DETR (it typically needs lower threshold)
rtdetr_threshold = st.sidebar.slider(
    "RT-DETRv2 Specific Threshold", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.10, 
    step=0.05,
    help="RT-DETRv2 was trained with limited data and may output lower confidence scores."
)

# Load Models
@st.cache_resource
def load_detector(yolo_p, rtdetr_p, custom_ocr_p=None):
    if inference is None:
        return None
    
    # Only pass paths if they exist, otherwise pass None to avoid crashes in init
    yp = yolo_p if os.path.exists(yolo_p) else None
    rp = rtdetr_p if os.path.exists(rtdetr_p) else None
    ocr_p = custom_ocr_p if (custom_ocr_p and os.path.exists(custom_ocr_p)) else None
    
    if yp is None and rp is None:
        return None

    return inference.LicensePlateDetector(yp, rp, custom_ocr_path=ocr_p)

# Initialize Detector
detector = None
if st.sidebar.button("Load / Reload Models"):
    if not (yolo_exists or rtdetr_exists):
        st.error("Cannot load models: Neither file path exists.")
    else:
        with st.spinner("Loading models... (This may take a moment)"):
            # Only use custom OCR if toggle is enabled AND path exists
            ocr_path_to_use = custom_ocr_path if (use_custom_ocr and custom_ocr_exists) else None
            detector = load_detector(yolo_path, rtdetr_path, ocr_path_to_use)
            if detector:
                st.sidebar.success("Models Loaded!")
                if detector.ocr_mode == "custom":
                    st.sidebar.success("✓ Using fine-tuned OCR")
                else:
                    st.sidebar.info("ℹ️ Using EasyOCR")
            else:
                 st.sidebar.error("Failed to initialize detector.")

# Try to load if already cached and we haven't explicitly reloaded
if detector is None and (yolo_exists or rtdetr_exists):
    ocr_path_to_use = custom_ocr_path if (use_custom_ocr and custom_ocr_exists) else None
    detector = load_detector(yolo_path, rtdetr_path, ocr_path_to_use)

# Tabs for Mode Selection
tab1, tab2 = st.tabs(["📷 Image Upload", "✍️ Direct Input"])

with tab1:
    uploaded_file = st.file_uploader("Upload Vehicle Image", type=['jpg', 'png', 'jpeg'])

    # Reset stored detections when a new file is uploaded
    uploaded_name = uploaded_file.name if uploaded_file is not None else None
    if st.session_state.get("_last_uploaded_name") != uploaded_name:
        st.session_state["_last_uploaded_name"] = uploaded_name
        st.session_state.pop("results_yolo", None)
        st.session_state.pop("results_rtdetr", None)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display Original Image
        st.image(image, caption="Uploaded Image", width=400)

        if st.button("Run Detection", type="primary"):
            if not detector:
                st.error("Models not loaded! Check sidebar paths.")
            else:
                with st.spinner("Running detection..."):
                    st.session_state["results_yolo"] = (
                        detector.predict_yolo(image, conf_threshold=conf_threshold)
                        if detector.yolo_model
                        else []
                    )
                    # Use separate RT-DETR threshold (typically needs lower value)
                    st.session_state["results_rtdetr"] = (
                        detector.predict_rtdetr(image, score_threshold=rtdetr_threshold)
                        if detector.rtdetr_model
                        else []
                    )

        # Always render the last computed detections (Streamlit reruns on every widget interaction)
        results_a = st.session_state.get("results_yolo", [])
        results_b = st.session_state.get("results_rtdetr", [])

        col1, col2 = st.columns(2)

        # --- Pipeline A: YOLOv8 ---
        with col1:
            st.header("Pipeline A (YOLOv8)")
            if not detector or not detector.yolo_model:
                st.warning("YOLO model not loaded.")
            elif not results_a:
                st.info("No detections yet. Click 'Run Detection'.")
            else:
                for idx, res in enumerate(results_a):
                    st.image(res['crop'], caption=f"Detection {idx+1}")
                    st.info(f"raw OCR: {res['text']}")

                    corrected_text = st.text_input(
                        f"Correct reading A-{idx}",
                        value=res['text'],
                        key=f"yolo_{idx}",
                    )
                    st.write(f"**Final:** {corrected_text}")

        # --- Pipeline B: RT-DETRv2 ---
        with col2:
            st.header("Pipeline B (RT-DETRv2)")
            if not detector or not detector.rtdetr_model:
                st.warning("RT-DETR model not loaded.")
            elif not results_b:
                st.info("No detections yet. Click 'Run Detection'.")
            else:
                for idx, res in enumerate(results_b):
                    st.image(res['crop'], caption=f"Detection {idx+1}")
                    st.info(f"raw OCR: {res['text']}")

                    corrected_text = st.text_input(
                        f"Correct reading B-{idx}",
                        value=res['text'],
                        key=f"rtdetr_{idx}",
                    )
                    st.write(f"**Final:** {corrected_text}")

        st.divider()
        st.subheader("🚗 Banten Vehicle Tax Inquiry")

        # Build selectable candidates from BOTH pipelines, using the current corrected text inputs.
        candidates: List[Tuple[str, str]] = []  # (label, plate_text)

        if results_a:
            for idx, _res in enumerate(results_a):
                current_text = st.session_state.get(f"yolo_{idx}", _res.get("text", ""))
                if current_text and current_text != "OCR_ERR":
                    candidates.append((f"Pipeline A (YOLOv8) — Detection {idx+1}: {current_text}", current_text))

        if results_b:
            for idx, _res in enumerate(results_b):
                current_text = st.session_state.get(f"rtdetr_{idx}", _res.get("text", ""))
                if current_text and current_text != "OCR_ERR":
                    candidates.append((f"Pipeline B (RT-DETRv2) — Detection {idx+1}: {current_text}", current_text))

        if not candidates:
            st.info("No plate text available yet. Run detection first, then correct OCR if needed.")
        else:
            selected_label = st.selectbox(
                "Select which detected plate to query",
                options=[c[0] for c in candidates],
                key="tax_plate_selector",
            )
            selected_text = next((t for (lbl, t) in candidates if lbl == selected_label), "")

            # Update the text input value when dropdown selection changes
            if "last_selected_plate" not in st.session_state:
                st.session_state.last_selected_plate = selected_label
                st.session_state.tax_plate_from_detection = selected_text
            elif st.session_state.last_selected_plate != selected_label:
                st.session_state.last_selected_plate = selected_label
                st.session_state.tax_plate_from_detection = selected_text

            plate_to_query = st.text_input(
                "Plate to query (editable)",
                key="tax_plate_from_detection",
                help="You can edit the plate text before querying the Banten database.",
            )

            if st.button("Query Tax Information", key="tax_query_from_detection"):
                with st.spinner("Querying Banten database..."):
                    result = check_banten_tax_live(plate_to_query)

                if result.get("Status") == "Success":
                    st.success(f"Success: {result.get('Queried', '')}")
                    st.table(result.get("Data", {}))
                elif result.get("Status") == "Not Found":
                    st.warning(result.get("Message", "Not found."))
                else:
                    st.error(result.get("Message", "Error"))
                    detail = result.get("Detail")
                    if detail:
                        st.code(detail)

with tab2:
    st.header("Direct Text Input")
    plate_text = st.text_input("Enter Plate Number manually (e.g. B 1234 CD)")
    if plate_text:
        st.success(f"Processing Plate: {plate_text.upper()}")
        if st.button("Query Tax Information", key="tax_query_manual"):
            with st.spinner("Querying Banten database..."):
                result = check_banten_tax_live(plate_text)

            if result.get("Status") == "Success":
                st.success(f"Success: {result.get('Queried', '')}")
                st.table(result.get("Data", {}))
            elif result.get("Status") == "Not Found":
                st.warning(result.get("Message", "Not found."))
            else:
                st.error(result.get("Message", "Error"))
                detail = result.get("Detail")
                if detail:
                    st.code(detail)
