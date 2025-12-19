import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import streamlit as st
import sys

st.write(f"Python Version: {sys.version}")

def check_import(module_name):
    try:
        __import__(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)

modules = [
    "cv2",
    "numpy",
    "easyocr",
    "torch",
    "ultralytics",
    "transformers",
    "PIL"
]

results = {}
for mod in modules:
    results[mod] = check_import(mod)

with open("deps_status.txt", "w") as f:
    f.write(f"Python Version: {sys.version}\n")
    for mod, (success, msg) in results.items():
        f.write(f"{mod}: {'INSTALLED' if success else 'MISSING'} - {msg}\n")
        if success:
            st.success(f"{mod}: INSTALLED")
        else:
            st.error(f"{mod}: MISSING - {msg}")

st.write("Check complete.")
