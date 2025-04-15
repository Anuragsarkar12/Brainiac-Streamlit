
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
import pandas as pd

# Check if GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu").to(device)


def load_lottie_url(ur: str):
    response=requests.get(lottie_url)
    if response.status_code !=200:
        return None
    return response.json()

lottie_url = "https://lottie.host/de2be76c-d3bf-467a-a69c-64fbdd9b8de4/i1ZZxc6oiV.json"
lottie_animation=load_lottie_url(lottie_url)
        
# Load the CNN model for tumor classification
cnn_model = load_model('model/buildspace_tumor_classifier.h5')
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def process_image(image):
    # Reduce image resolution to speed up processing (optional)
    new_height = min(720, image.height)
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_height % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    
    # Resize RGB image
    image = image.resize(new_size)

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    pad = 16
    output = predicted_depth.squeeze().cpu().numpy()
    output = output[pad:-pad, pad:-pad]
    output = cv2.GaussianBlur(output, (5, 5), 0)
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # Resize depth image to match RGB image size
    depth_image = cv2.resize(output, (image.width, image.height))

    depth_image = (depth_image * 255 / np.max(depth_image)).astype('uint8')
    image = np.array(image)

    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(image.shape[1], image.shape[0], 500, 500, image.shape[1] / 2, image.shape[0] / 2)

    # Optimize Open3D point cloud creation and processing
    pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)  # Adjusted for speed
    pcd = pcd_raw.select_by_index(ind)

    # Estimate and orient normals efficiently
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    return pcd

def classify_tumor(image):
    # Resize image to match input size of the CNN model
    image = image.resize((128, 128))
    
    # Convert image to array and normalize
    image_array = np.array(image) / 255.0
    
    # Expand dimensions to match the input shape of the model (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict using the CNN model
    prediction = cnn_model.predict(image_array)
    
    # Get the class labels
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Get the predicted class
    predicted_class = class_labels[np.argmax(prediction)]
    
    return predicted_class

def main():
    st.set_page_config(
        page_title="Brainiac",
        page_icon=":brain:",
        layout="wide"
    )

    gif_animation=''

    col1,col2=st.columns(2)
    with col1:
        st.markdown(
            "<h1 style='animation: fadein 1s ease-in;'>Brainiac</h1>",
            unsafe_allow_html=True,
        )
        description='''Brainiac is a web application designed to help users visualize and classify brain MRI scans.'''
        st.markdown(
            description,
            unsafe_allow_html=True,
        )

        with col2:
            if lottie_animation:
                with st.container():
                    st_lottie(lottie_animation,height=140, width=140)
            else:
                st.write("Failed to load Lottie Animation")


        with st.sidebar:
            uploaded_file=st.file_uploader("Upload your MRI Scan...",type=["jpg","jpeg","png"])
            classify_button=st.button("Classify")
            visualize_button=st.button("Visualize")


        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image,caption="Uploaded MRI Scan", use_column_width=True)


            if visualize_button:
                with st.spinner("Processing"):
                    pcd=process_image(image)
                    vis=o3d.visualization.Visualizer()
                    vis.create_window(width=800, height=800)
                    vis.add_geometry(pcd)
                    vis.get_render_option().point_size=2
                    vis.run()
                    vis.destroy_window()

            if classify_button:
                with st.spinner("Classifying..."):
                    predicted_class = classify_tumor(image)
                st.success("Classification Complete!")
                st.header("Classification Result:")
                st.write(f"Tumor Classification: {predicted_class}")


        

if __name__ == '__main__':
    main()
