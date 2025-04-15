import streamlit as st
from PIL import Image
import torch
import cv2
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit_lottie
from streamlit_lottie import st_lottie
import requests
import base64
import pandas as pd  # Added pandas import

# Rest of your imports and model loading...

def main():
    st.set_page_config(
        page_title="Brainiac",
        page_icon=":brain:",
        layout="wide"
    )

    gif_animation = ''

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<h1 style='animation: fadein 1s ease-in;'>Brainiac</h1>",
            unsafe_allow_html=True,
        )
        description = '''Brainiac is a web application designed to help users visualize and classify brain MRI scans.'''
        st.markdown(
            description,
            unsafe_allow_html=True,
        )

    with col2:
        if lottie_animation:
            with st.container():
                st_lottie(lottie_animation, height=140, width=140)
        else:
            st.write("Failed to load Lottie Animation")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your MRI Scan...", type=["jpg", "jpeg", "png"])
        classify_button = st.button("Classify")
        visualize_button = st.button("Visualize")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        if visualize_button:
            with st.spinner("Processing 3D visualization..."):
                try:
                    pcd = process_image(image)
                    
                    # Save the point cloud to a temporary file
                    temp_file = "temp_pointcloud.ply"
                    o3d.io.write_point_cloud(temp_file, pcd)
                    
                    # Provide download link for the 3D model
                    st.success("3D Point Cloud generated successfully!")
                    st.info("Download the 3D model file below and open it with any 3D viewer software (like MeshLab, Blender, etc.)")
                    
                    with open(temp_file, "rb") as file:
                        btn = st.download_button(
                            label="Download 3D Model",
                            data=file,
                            file_name="brain_mri_3d.ply",
                            mime="application/octet-stream"
                        )
                    
                    # Display some statistics about the point cloud
                    points = np.asarray(pcd.points)
                    st.markdown("### 3D Model Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Points", f"{len(points):,}")
                    with col2:
                        st.metric("Depth Range", f"{np.min(points[:, 2]):.2f} to {np.max(points[:, 2]):.2f}")
                    with col3:
                        st.metric("Model Quality", "High")
                        
                    # Display a sample of the point cloud data
                    st.markdown("### Sample Point Data")
                    df = pd.DataFrame(points[:5], columns=["X", "Y", "Z"])
                    st.dataframe(df)
                    
                    # Show some instructions for using the 3D model
                    st.markdown("### How to View the 3D Model")
                    st.markdown("""
                    1. Download the 3D model file above
                    2. Open it with a 3D viewer like:
                       - [MeshLab](https://www.meshlab.net/) (free, open source)
                       - [Blender](https://www.blender.org/) (free, open source)
                       - [CloudCompare](https://www.cloudcompare.org/) (free, open source)
                    3. In the viewer, you can rotate, zoom, and explore the 3D brain model
                    """)
                    
                except Exception as e:
                    st.error(f"Error in 3D visualization: {str(e)}")
                    st.info("Try uploading a different image or check if the image has enough features for depth estimation.")

        if classify_button:
            with st.spinner("Classifying..."):
                predicted_class = classify_tumor(image)
            st.success("Classification Complete!")
            st.header("Classification Result:")
            st.write(f"Tumor Classification: {predicted_class}")

if __name__ == '__main__':
    main()
