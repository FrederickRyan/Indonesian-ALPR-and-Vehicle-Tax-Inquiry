import cv2
import numpy as np
import os
import json
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None
    print("Warning: easyocr module not found.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None
    print("Warning: torch module not found.")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics module not found.")

try:
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
except ImportError:
    RTDetrV2ForObjectDetection = None
    RTDetrImageProcessor = None
    print("Warning: transformers module not found.")

# Try to import custom OCR module
try:
    from custom_plate_ocr import IndonesianPlateOCR
except ImportError:
    IndonesianPlateOCR = None
    print("Note: custom_plate_ocr module not found, will use EasyOCR only.")


class LicensePlateDetector:
    def __init__(self, yolo_path=None, rtdetr_path=None, custom_ocr_path=None):
        if torch:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize OCR - try custom model first, fallback to EasyOCR
        self.custom_ocr = None
        self.reader = None  # EasyOCR reader
        self.ocr_mode = "none"
        
        # Try to load custom fine-tuned OCR if path provided
        if custom_ocr_path and os.path.exists(custom_ocr_path) and IndonesianPlateOCR is not None:
            print("Loading fine-tuned Indonesian Plate OCR...")
            try:
                self.custom_ocr = IndonesianPlateOCR(custom_ocr_path, device=self.device)
                if self.custom_ocr.is_loaded:
                    self.ocr_mode = "custom"
                    print("✓ Using fine-tuned Indonesian Plate OCR")
            except Exception as e:
                print(f"Custom OCR failed: {e}, falling back to EasyOCR")
        
        # Fallback to EasyOCR if custom not available
        if self.ocr_mode != "custom":
            print("Initializing EasyOCR...")
            if easyocr:
                try:
                    self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
                    self.ocr_mode = "easyocr"
                    print("✓ Using EasyOCR")
                except Exception as e:
                    print(f"Warning: EasyOCR failed to initialize: {e}")
                    self.reader = None
            else:
                print("Warning: EasyOCR not available.")
                self.reader = None
        
        # Load detection models if paths provided
        self.yolo_model = None
        self.rtdetr_model = None
        self.rtdetr_processor = None
        
        if yolo_path:
            self.load_yolo(yolo_path)
            
        if rtdetr_path:
            self.load_rtdetr(rtdetr_path)
            
    def load_yolo(self, path):
        if not os.path.exists(path):
            print(f"Error: YOLO path not found at {path}")
            return

        try:
            print(f"Loading YOLO from {path}")
            if YOLO:
                self.yolo_model = YOLO(path)
            else:
                 print("Warning: YOLO module not available.")
                 self.yolo_model = None
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            self.yolo_model = None

    def load_rtdetr(self, path):
        if not os.path.exists(path):
            print(f"Error: RT-DETR path not found at {path}")
            return

        try:
            print(f"Loading RT-DETR from {path}")
            if RTDetrImageProcessor and RTDetrV2ForObjectDetection:
                self.rtdetr_processor = RTDetrImageProcessor.from_pretrained(path)
                self.rtdetr_processor.do_resize = True
                self.rtdetr_processor.size = {"height": 640, "width": 640}
                
                self.rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained(
                    path, 
                    num_labels=1, 
                    ignore_mismatched_sizes=True
                ).to(self.device)
                self.rtdetr_model.eval() # Set to eval mode for inference
            else:
                print("Warning: Transformers module not available.")
                self.rtdetr_model = None
                self.rtdetr_processor = None
        except Exception as e:
            print(f"Error loading RT-DETR: {e}")
            self.rtdetr_model = None

    def perform_ocr(self, image_np):
        """
        Run OCR on a numpy image crop.
        Uses fine-tuned model if available, otherwise falls back to EasyOCR.
        """
        if image_np is None or image_np.size == 0:
            return ""
        
        # Try custom fine-tuned OCR first
        if self.ocr_mode == "custom" and self.custom_ocr and self.custom_ocr.is_loaded:
            try:
                text = self.custom_ocr.recognize(image_np)
                if text:
                    # Clean and format the text
                    text = text.upper()
                    text = ''.join(c for c in text if c.isalnum() or c == ' ')
                    return text.strip()
            except Exception as e:
                print(f"Custom OCR error: {e}")
        
        # Fallback to EasyOCR
        if self.reader is None:
            return "OCR_ERR"

        try:
            results = self.reader.readtext(image_np, detail=0, paragraph=True)
            text = " ".join(results).upper()
            # Filter alphanumeric
            text = ''.join(c for c in text if c.isalnum() or c == ' ')
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def crop_plate(self, original_image, box):
        """
        Crop image based on bbox [x1, y1, x2, y2]
        """
        h, w, _ = original_image.shape
        x1, y1, x2, y2 = map(int, box)
        
        # Clamp coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return None
            
        return original_image[y1:y2, x1:x2]

    def predict_yolo(self, image, conf_threshold=0.25):
        """
        Returns list of dicts: {'box': [x1,y1,x2,y2], 'text': '...', 'conf': float}
        """
        if not self.yolo_model:
            return []

        # Run inference
        results = self.yolo_model.predict(image, verbose=False, conf=conf_threshold)
        final_results = []
        
        # Convert PIL to Numpy (RGB)
        img_np = np.array(image)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
                conf = float(box.conf[0])
                
                crop = self.crop_plate(img_np, coords)
                text = self.perform_ocr(crop)
                
                final_results.append({
                    "box": coords.tolist(),
                    "text": text,
                    "conf": conf,
                    "crop": crop
                })
        
        return final_results

    def predict_rtdetr(self, image, score_threshold=0.45):
        """
        Returns list of dicts: {'box': [x1,y1,x2,y2], 'text': '...', 'conf': float}
        """
        if not self.rtdetr_model or not self.rtdetr_processor:
            return []

        # Preprocess
        inputs = self.rtdetr_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.rtdetr_model(**inputs)
        
        # Post-process
        # Need target sizes for converting absolute coordinates
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device) # (h, w)
        
        # Use configurable threshold
        results = self.rtdetr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]
        
        final_results = []
        img_np = np.array(image)
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy() # x1, y1, x2, y2
            score = float(score)
            
            crop = self.crop_plate(img_np, box)
            text = self.perform_ocr(crop)
            
            final_results.append({
                "box": box.tolist(),
                "text": text,
                "conf": score,
                "crop": crop
            })
            
        return final_results
