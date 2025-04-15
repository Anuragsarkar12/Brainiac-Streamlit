import streamlit as st
from PIL import Image
import torch
import cv2
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit_lottie
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import pandas as pd

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load depth estimation model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu").to(device)

# Load lottie animation
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

lottie_url = "https://lottie.host/de2be76c-d3bf-467a-a69c-64fbdd9b8de4/i1ZZxc6oiV.json"
lottie_animation = load_lottie_url(lottie_url)

# Load CNN model
cnn_model = load_model('model/buildspace_tumor_classifier.h5')
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Process image and convert to 3D point cloud for Plotly
def process_image_for_plotly(image):
    new_height = min(720, image.height)
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_height % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    
    image = image.resize(new_size)
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    pad = 16
    depth = predicted_depth.squeeze().cpu().numpy()
    depth = depth[pad:-pad, pad:-pad]
    depth = cv2.GaussianBlur(depth, (5, 5), 0)

    image = image.crop((pad, pad, image.width - pad, image.height - pad))
    image_np = np.array(image)
    depth_resized = cv2.resize(depth, (image_np.shape[1], image_np.shape[0]))

    # Normalize depth
    depth_resized = (depth_resized - np.min(depth_resized)) / (np.max(depth_resized) - np.min(depth_resized))

    # Create 3D coordinates
    h, w = depth_resized.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_resized

    # Flatten for plotly
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    colors = image_np.reshape(-1, 3)

    return x, y, z, colors

# Classify tumor type
def classify_tumor(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = cnn_model.predict(image_array)
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

# Plotly scatter3D for RGB point cloud
def display_3d_plot(x, y, z, colors):
    color_hex = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=color_hex,
            opacity=0.8
        )
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Depth'
    ), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

# Main app logic
def main():
    st.set_page_config(
        page_title="Brainiac",
        page_icon=":brain:",
        layout="wide"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h1 style='animation: fadein 1s ease-in;'>Brainiac</h1>", unsafe_allow_html=True)
        description = "Brainiac is a web application designed to help users visualize and classify brain MRI scans."
        st.markdown(description, unsafe_allow_html=True)

    with col2:
        if lottie_animation:
            st_lottie(lottie_animation, height=140, width=140)
        else:
            st.write("Failed to load Lottie animation")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your MRI Scan...", type=["jpg", "jpeg", "png"])
        classify_button = st.button("Classify")
        visualize_button = st.button("Visualize")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        if visualize_button:
            with st.spinner("Creating 3D Visualization..."):
                x, y, z, colors = process_image_for_plotly(image)
                display_3d_plot(x, y, z, colors)

        if classify_button:
            with st.spinner("Classifying..."):
                predicted_class = classify_tumor(image)
            st.success("Classification Complete!")
            st.header("Classification Result:")
            st.write(f"Tumor Classification: **{predicted_class}**")

if __name__ == '__main__':
    main()
