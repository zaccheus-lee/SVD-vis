import streamlit as st
import numpy as np
from PIL import Image

from src.image_processing import compress_image, normalize_image

st.title("SVD Image Compression")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Singular value selection
    k = st.slider("Number of Singular Values", 1, 100, 10)
    
    # Compress image
    compressed_image = compress_image(image, k)
    compressed_image_normalized = normalize_image(compressed_image)
    compressed_image_pil = Image.fromarray(compressed_image_normalized)
    
    # Display original and compressed images
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    with col2:
        st.image(compressed_image_pil, caption=f"Compressed Image (k={k})", use_container_width=True) 