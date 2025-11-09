import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

from src.image_processing import compress_image, normalize_image

st.title("SVD Image Compression")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Convert to RGB if it's a different mode, or keep as grayscale
    if image.mode not in ['RGB', 'L']:
        image = image.convert('RGB')
    
    # Display original image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_container_width=True)
    
    # Singular value selection
    k = st.slider("Number of Singular Values", 1, 100, 5)
    
    # Convert image to array and perform SVD on each channel
    img_array = np.array(image)
    
    # Perform SVD to get U, Sigma, VT matrices
    st.subheader("SVD Decomposition Matrices")
    
    # For display purposes, use grayscale version or first channel
    if len(img_array.shape) == 2:
        display_channel = img_array
    else:
        display_channel = img_array[:, :, 0]  # Use first channel (Red)
    
    U, S, Vt = np.linalg.svd(display_channel, full_matrices=False)
    
    # Display matrices for each k value
    for i in range(k):
        with st.expander(f"Matrices for k={i+1}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**U[:, {i}]** (Left Singular Vector)")
                st.write(f"Shape: {U[:, i].shape}")
                # Display first 10 values
                u_display = pd.DataFrame(U[:min(10, len(U)), i], columns=[f"U[:, {i}]"])
                st.dataframe(u_display, use_container_width=True)
                if len(U) > 10:
                    st.caption(f"... showing first 10 of {len(U)} values")
            
            with col2:
                st.write(f"**σ_{i+1}** (Singular Value)")
                st.write(f"Value: {S[i]:.6f}")
                st.write(f"As diagonal matrix element: [[{S[i]:.6f}]]")
            
            with col3:
                st.write(f"**V^T[{i}, :]** (Right Singular Vector)")
                st.write(f"Shape: {Vt[i, :].shape}")
                # Display first 10 values
                vt_display = pd.DataFrame(Vt[i, :min(10, Vt.shape[1])].reshape(1, -1), 
                                         columns=[f"V^T[{i}, {j}]" for j in range(min(10, Vt.shape[1]))])
                st.dataframe(vt_display, use_container_width=True)
                if Vt.shape[1] > 10:
                    st.caption(f"... showing first 10 of {Vt.shape[1]} values")
            
            # Show the rank-1 approximation formula
            st.write(f"**Rank-1 Matrix: σ_{i+1} × U[:, {i}] × V^T[{i}, :]**")
            rank_one = S[i] * np.outer(U[:, i], Vt[i, :])
            st.write(f"Resulting matrix shape: {rank_one.shape}")
            # Show a small sample of the resulting matrix
            sample_size = min(5, rank_one.shape[0]), min(5, rank_one.shape[1])
            rank_one_sample = pd.DataFrame(rank_one[:sample_size[0], :sample_size[1]])
            st.write("Sample (top-left corner):")
            st.dataframe(rank_one_sample, use_container_width=True)
    
    # Store individual components for each k value
    st.subheader(f"Individual Singular Value Components (k=1 to k={k})")
    
    # Display individual components in a grid
    cols_per_row = 4
    for i in range(k):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        # Compress image using only the (i+1)th singular value
        component_image = compress_image(image, i + 1)
        
        # If i > 0, subtract the previous sum to get only this component
        if i > 0:
            previous_sum = compress_image(image, i)
            component_only = component_image - previous_sum
        else:
            component_only = component_image
        
        # Normalize and display
        component_normalized = normalize_image(component_only)
        
        # Handle grayscale vs color images
        if len(component_normalized.shape) == 2:
            component_pil = Image.fromarray(component_normalized, mode='L')
        else:
            component_pil = Image.fromarray(component_normalized)
        
        with cols[i % cols_per_row]:
            st.image(component_pil, caption=f"k={i+1}", use_container_width=True)
    
    # Display cumulative sums
    st.subheader(f"Cumulative Reconstruction (Sum of Components)")
    
    for i in range(k):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        # Compress image using sum of first (i+1) components
        cumulative_image = compress_image(image, i + 1)
        cumulative_normalized = normalize_image(cumulative_image)
        
        # Handle grayscale vs color images
        if len(cumulative_normalized.shape) == 2:
            cumulative_pil = Image.fromarray(cumulative_normalized, mode='L')
        else:
            cumulative_pil = Image.fromarray(cumulative_normalized)
        
        with cols[i % cols_per_row]:
            st.image(cumulative_pil, caption=f"Sum k=1 to k={i+1}", use_container_width=True)
    
    # Final compressed image
    st.subheader(f"Final Compressed Image (k={k})")
    compressed_image = compress_image(image, k)
    compressed_image_normalized = normalize_image(compressed_image)
    
    # Handle grayscale vs color images
    if len(compressed_image_normalized.shape) == 2:
        compressed_image_pil = Image.fromarray(compressed_image_normalized, mode='L')
    else:
        compressed_image_pil = Image.fromarray(compressed_image_normalized)
    
    st.image(compressed_image_pil, caption=f"Final Image (k={k})", use_container_width=True) 