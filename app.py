import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import time
from torchvision import transforms
import sys
import os

# This tells Python to look inside the 'src' folder for sod_model.py
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sod_model import SODModel

# Page Config
st.set_page_config(page_title="Salient Object Detection", layout="wide")
st.title("🎯 Salient Object Detector")
st.write("Upload an image to see the model identify the most important object.")

# 1. Load the Model
@st.cache_resource # Keeps the model in memory so it doesn't reload every time
def load_model():
    model = SODModel()
    # Load weights - ensure the file name matches exactly
    model.load_state_dict(torch.load("best_sod_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 2. Image Preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 3. Sidebar / Upload
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Input Image
    img = Image.open(uploaded_file).convert("RGB")
    
    # Run Inference
    input_tensor = preprocess(img)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000 # Convert to ms

    # Process Output
    mask = output.squeeze().cpu().numpy()
    mask_img = (mask * 255).astype(np.uint8)
    
    # Create Overlay
    # Resize mask back to original image size for overlay
    mask_resized = Image.fromarray(mask_img).resize(img.size)
    overlay = np.array(img).copy()
    mask_np = np.array(mask_resized)
    # Apply green tint where mask is high
    overlay[mask_np > 127] = [0, 255, 0] # Green overlay

    # --- DISPLAY RESULTS ---
    st.success(f"Inference Time: {inference_time:.2f} ms")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Input Image")
        st.image(img, use_container_width=True)
        
    with col2:
        st.header("Saliency Mask")
        st.image(mask_img, use_container_width=True)
        
    with col3:
        st.header("Overlay Result")
        st.image(overlay, use_container_width=True)
else:
    st.info("Please upload an image in the sidebar to begin.")