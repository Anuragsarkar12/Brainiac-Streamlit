import streamlit as st


def main():
    
    st.set_page_config(
        page_title="documentation",
        page_icon=":book:",
        layout="wide"
    )

    # Define the main content layout
    st.title("Brainiac Web App Documentation")
    st.write("Welcome to the documentation for Brainiac, a web app for MRI scan visualization and tumor classification.")

    st.header("Overview")
    st.markdown("""
    Brainiac is a web application designed to assist users in visualizing and classifying brain MRI scans.
    """)

    st.header("Features")
    st.markdown("""
    - **MRI Scan Upload:** Users can upload MRI scans in JPEG, JPG, or PNG formats.
    - **Visualization:** Provides a 3D point cloud visualization of the MRI scan using Open3D.
    - **Tumor Classification:** Utilizes a CNN model to classify tumors into specific types.
    """)

    st.header("How to Use Brainiac")
    st.markdown("""
    1. **Uploading an MRI Scan:**
       - Click on the "Upload your MRI Scan..." button in the sidebar.
       - Select a JPEG, JPG, or PNG file containing the MRI scan.

    2. **Visualization:**
       - Click on the "Visualize" button to see a 3D point cloud representation of the uploaded MRI scan.

    3. **Tumor Classification:**
       - Click on the "Classify" button to classify whether the MRI scan shows any tumor.
       - The classification result will appear on the screen.
    """)

    st.header("Technical Details")
    st.markdown("""
    - **Machine Learning Model:** Brainiac uses a CNN model trained to classify brain tumors.
    - **Visualization Tool:** Open3D is used to render and manipulate the 3D point cloud.
    - **Framework:** Built using Streamlit, a Python framework for creating web applications.
    """)

    st.header("Requirements")
    st.markdown("""
    - Ensure your MRI scan is in JPEG, JPG, or PNG format.
    - A modern web browser that supports WebGL for 3D visualization.
    """)

    st.header("Troubleshooting")
    st.markdown("""
    - **Failed Uploads:** Ensure your file is in the correct format (JPEG, JPG, or PNG).
    - **Visualization Issues:** Check your internet connection and WebGL compatibility.
    """)



if __name__ == '__main__':
    main()
