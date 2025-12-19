
import os
import sys
import math
import contextlib
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

def patched_infer(model, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
    # Dynamically get helpers from the model's module
    m = sys.modules[model.__module__]
    format_messages = m.format_messages
    load_pil_images = m.load_pil_images
    BasicImageTransform = m.BasicImageTransform
    text_encode = m.text_encode
    dynamic_preprocess = m.dynamic_preprocess
    
    torch_dtype = model.dtype
    device = model.device

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)

    # Construct conversation
    if prompt and image_file:
        conversation = [
            {
                "role": "<|User|>",
                "content": f'{prompt}',
                "images": [f'{image_file}'],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    elif prompt:
         conversation = [
            {"role": "<|User|>", "content": f'{prompt}'},
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        assert False, f'prompt is none!'
    
    # Preprocessing
    prompt_text = format_messages(conversations=conversation, sft_format='plain', system_prompt='')
    patch_size = 16
    downsample_ratio = 4
    images = load_pil_images(conversation)
    ratio = 1
    image_draw = images[0].copy()
    w,h = image_draw.size
    ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    images_seq_mask = []
    image_token = '<image>'
    image_token_id = 128815
    text_splits = prompt_text.split(image_token)

    images_list, images_crop_list, images_seq_mask = [], [], []
    tokenized_str = []
    images_spatial_crop = []
    
    for text_sep, image in zip(text_splits, images):
        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
                images_crop_raw = [] # No crops if small? Wait, existing logic might differ. 
                # Checking logic: if size <= 640, crop_ratio=[1,1].
                # logic for crops:
                # if crop_mode: dynamic_preprocess...
                # existing logic has indentation that implies dynamic_preprocess is skipped if condition met?
                # Actually, let's just stick to dynamic_preprocess as mostly safe default or copy specific logic if crucial.
                # Simplification: Always use dynamic_preprocess if unsure, but let's follow standard flow.
                pass 
            
            # Re-implementing logic flow from source:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
                images_crop_raw = [] # Assuming no crops needed or handled differently?
                # Actually usage of images_crop_raw happens later.
                # If we look at source (lines 784-793):
                # if small: crop_ratio=[1,1].
                # else: dynamic_preprocess(...) -> returns processed_images, ratio
                pass
            else:
                 images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=image_size)

            # Global View
            global_view = ImageOps.pad(image, (base_size, base_size), color=tuple(int(x * 255) for x in image_transform.mean))
            
            # Token counting logic (simplified)
            if base_size == 1024: valid_img_tokens = int(256 * ratio) # ... (variable unused for inference really)

            images_list.append(image_transform(global_view).to(torch_dtype).to(device))
            
            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(image_transform(images_crop_raw[i]).to(torch_dtype).to(device))
            
            # Add image tokens
            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
            
            tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

    # Process last split
    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    # BOS
    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str 
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

    # Stack Images
    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size)).to(torch_dtype).to(device)
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long).to(device)
        images_crop = torch.zeros((1, 3, base_size, base_size)).to(torch_dtype).to(device)
    else:
        images_ori = torch.stack(images_list, dim=0).to(device)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long).to(device)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0).to(device)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size)).to(torch_dtype).to(device)

    # Generate
    # We use eval_mode=True logic
    with torch.no_grad():
        # Autocast if cuda, but we are fixing for CPU possibility too. 
        # If cpu, calling autocast('cuda') is harmless or ignored, but let's be safe.
        ctx = torch.autocast("cuda", dtype=torch_dtype) if "cuda" in str(device) else contextlib.nullcontext()
        with ctx:
            output_ids = model.generate(
                input_ids.unsqueeze(0).to(device),
                images=[(images_crop, images_ori)],
                images_seq_mask = images_seq_mask.unsqueeze(0).to(device),
                images_spatial_crop = images_spatial_crop,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=64, # Reduced from 8192 for Plate OCR on CPU
                do_sample=False,   # Greedy decoding
                use_cache = True
            )

    # Decode
    outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).shape[1]:])
    stop_str = '<｜end▁of▁sentence｜>'
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


class IndonesianPlateOCR:
    def __init__(self, lora_path, base_model_id="unsloth/DeepSeek-OCR", device="cuda"):
        self.device = device
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        
        print("-" * 50)
        print(f"Initializing DeepSeek-OCR...")
        print(f"Base Model: {base_model_id}")
        print(f"Adapter:    {lora_path}")
        print("-" * 50)

        try:
            # 1. Check for bitsandbytes (Critical for 4GB VRAM)
            try:
                import bitsandbytes as bnb
                print("✓ bitsandbytes found (4-bit quantization available)")
                use_4bit = True
            except ImportError:
                print("! bitsandbytes NOT found. Model loading might fail on 4GB VRAM.")
                print("! Please install: pip install bitsandbytes")
                use_4bit = False

            # 2. Load Tokenizer
            print("Loading Tokenizer...")
            # Try loading from adapter first, else base
            if os.path.exists(os.path.join(lora_path, "tokenizer.json")):
                self.tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

            # 3. Load Model
            print("Loading Base Model (this may take time)...")
            from transformers import BitsAndBytesConfig
            
            bnb_config = None
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            self.model = AutoModel.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if use_4bit else "auto",
                device_map="auto"
            )

            # 4. Apply LoRA
            print(f"Applying LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()
            
            self.is_loaded = True
            print("✓ DeepSeek-OCR Loaded Successfully!")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Failed to load DeepSeek-OCR: {e}")
            print("Detailed Error Trace:")
            import traceback
            traceback.print_exc()
            self.is_loaded = False

    def recognize(self, image_input):
        """
        Predict text from an image.
        image_input: str path, PIL Image, or numpy array
        """
        if not self.is_loaded:
            print("Error: Model not loaded.")
            return ""

        try:
            # 1. Prepare Image & Save to Temp File (infer requires path)
            temp_image_path = "temp_ocr_input.jpg"
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
                temp_image_path = image_input # Use original path if valid
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert("RGB")
                image.save(temp_image_path)
            else:
                image = image_input
                image.save(temp_image_path)

            # 2. Prepare Prompt
            prompt = "<image>\nRead the Indonesian license plate number exactly as shown. "
            
            # 3. Inference using built-in method
            # PeftModel wraps the base model, so we access the base model's infer method
            # Depending on how Peft wraps, it might be model.model.infer or model.base_model.infer
            
            base_model = self.model
            if hasattr(self.model, 'base_model'):
                base_model = self.model.base_model
                if hasattr(base_model, 'model'): 
                     base_model = base_model.model

            # Use patched infer for device compatibility (avoids hardcoded .cuda())
            result = patched_infer(
                model=base_model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path="./temp_ocr_results",
                eval_mode=True
            )

            # Cleanup temp file if we created it
            if temp_image_path == "temp_ocr_input.jpg" and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
            return result.strip()

        except Exception as e:
            print(f"Recognition Error: {e}")
            import traceback
            traceback.print_exc()
            return ""
