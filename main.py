import streamlit as st
import cv2
import numpy as np
import os
import logging
from typing import Optional
import time

# Import custom modules
from edge_detectors import apply_sobel, apply_scharr, apply_laplacian, apply_canny, compare_all
from utils import preprocess_image, display_results, get_performance_notes, validate_image_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default image path
DEFAULT_IMAGE = os.path.join('images', 'sample.jpg')

def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the given file path
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        numpy.ndarray or None: Loaded image or None if loading fails
    """
    try:
        # Validate path
        if not validate_image_path(file_path):
            logger.error(f"Invalid image path: {file_path}")
            return None
        
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Failed to load image: {file_path}")
            return None
        
        logger.info(f"Successfully loaded image: {file_path}")
        return image
    
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None

def run_app():
    """
    Main function to run the application
    
    Key steps:
    1. Set up Streamlit UI with sidebar controls
    2. Load default or user-uploaded image
    3. Display original image
    4. Get user parameters for each edge detector
    5. Process image with selected detectors
    6. Display results in a grid layout
    7. Show optional performance notes
    """
    # Set page title and config
    st.set_page_config(page_title="Edge Vision", page_icon="🔍", layout="wide")
    st.title("Edge Vision: Visual Edge Detection Comparator")
    st.markdown("""
    This tool allows you to apply multiple edge detection algorithms to images and compare them side-by-side.
    Adjust the parameters in the sidebar to see how they affect the results.
    """)
    
    # Sidebar for controls
    st.sidebar.title("Settings")
    
    try:
        # Image Selection
        st.sidebar.header("Image")
        img_source = st.sidebar.radio("Select image source:", ["Default Image", "Upload Image"])
        
        if img_source == "Default Image":
            image = load_image(DEFAULT_IMAGE)
            if image is None:
                st.error(f"Failed to load default image. Please check that {DEFAULT_IMAGE} exists.")
                st.stop()
        else:
            uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    st.error("Failed to process the uploaded image.")
                    st.stop()
            else:
                st.info("Please upload an image or select 'Default Image'.")
                st.stop()
        
        # Image preprocessing
        max_width = st.sidebar.slider("Max image width", 300, 1200, 800)
        if image.shape[1] > max_width:
            height = int(image.shape[0] * (max_width / image.shape[1]))
            image = cv2.resize(image, (max_width, height))
        
        # Edge detector selection
        st.sidebar.header("Edge Detectors")
        show_sobel = st.sidebar.checkbox("Sobel", value=True)
        show_scharr = st.sidebar.checkbox("Scharr", value=True)
        show_laplacian = st.sidebar.checkbox("Laplacian", value=True)
        show_canny = st.sidebar.checkbox("Canny", value=True)
        
        # Parameter settings
        st.sidebar.header("Parameters")
        
        # Parameters for Sobel
        if show_sobel:
            st.sidebar.subheader("Sobel Parameters")
            sobel_ksize = st.sidebar.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3)
            sobel_scale = st.sidebar.slider("Scale", 1.0, 10.0, 1.0, 0.1)
            sobel_delta = st.sidebar.slider("Delta", 0.0, 255.0, 0.0, 1.0)
            sobel_dx = st.sidebar.slider("X Derivative Order", 0, 2, 1)
            sobel_dy = st.sidebar.slider("Y Derivative Order", 0, 2, 1)
        
        # Parameters for Scharr
        if show_scharr:
            st.sidebar.subheader("Scharr Parameters")
            scharr_scale = st.sidebar.slider("Scale (Scharr)", 1.0, 10.0, 1.0, 0.1)
            scharr_delta = st.sidebar.slider("Delta (Scharr)", 0.0, 255.0, 0.0, 1.0)
            scharr_dx = st.sidebar.radio("X Direction (Scharr)", [0, 1], 1)
            scharr_dy = st.sidebar.radio("Y Direction (Scharr)", [0, 1], 0)
        
        # Parameters for Laplacian
        if show_laplacian:
            st.sidebar.subheader("Laplacian Parameters")
            laplacian_ksize = st.sidebar.select_slider("Kernel Size (Laplacian)", options=[1, 3, 5, 7], value=3)
            laplacian_scale = st.sidebar.slider("Scale (Laplacian)", 1.0, 10.0, 1.0, 0.1)
            laplacian_delta = st.sidebar.slider("Delta (Laplacian)", 0.0, 255.0, 0.0, 1.0)
        
        # Parameters for Canny
        if show_canny:
            st.sidebar.subheader("Canny Parameters")
            canny_threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            canny_threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
            canny_aperture = st.sidebar.select_slider("Aperture Size", options=[3, 5, 7], value=3)
            canny_l2gradient = st.sidebar.checkbox("Use L2 Gradient", value=False)
        
        # Show performance notes
        show_notes = st.sidebar.checkbox("Show Performance Notes", value=False)
        
        # Process the image based on selections
        with st.spinner("Processing image..."):
            start_time = time.time()
            
            # Convert colors for display
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create a list to store results and titles
            results = [display_image]
            titles = ["Original"]
            
            # Apply selected edge detectors
            gray = preprocess_image(image)
            
            if show_sobel:
                sobel_result = apply_sobel(
                    gray, 
                    ksize=sobel_ksize, 
                    scale=sobel_scale, 
                    delta=sobel_delta, 
                    dx=sobel_dx, 
                    dy=sobel_dy
                )
                results.append(sobel_result)
                titles.append("Sobel")
            
            if show_scharr:
                scharr_result = apply_scharr(
                    gray, 
                    scale=scharr_scale, 
                    delta=scharr_delta, 
                    dx=scharr_dx, 
                    dy=scharr_dy
                )
                results.append(scharr_result)
                titles.append("Scharr")
            
            if show_laplacian:
                laplacian_result = apply_laplacian(
                    gray, 
                    ksize=laplacian_ksize, 
                    scale=laplacian_scale, 
                    delta=laplacian_delta
                )
                results.append(laplacian_result)
                titles.append("Laplacian")
            
            if show_canny:
                canny_result = apply_canny(
                    gray, 
                    threshold1=canny_threshold1, 
                    threshold2=canny_threshold2, 
                    aperture_size=canny_aperture, 
                    L2gradient=canny_l2gradient
                )
                results.append(canny_result)
                titles.append("Canny")
            
            # Display results
            if len(results) > 1:
                st.write(f"Processing completed in {time.time() - start_time:.2f} seconds")
                
                # Use Matplotlib to create a figure with all results
                fig = display_results(results, titles)
                st.pyplot(fig)
            else:
                st.warning("No edge detectors selected. Please select at least one from the sidebar.")
            
            # Show performance notes if requested
            if show_notes:
                st.header("Performance Notes")
                
                # Create tabs for each selected detector
                if any([show_sobel, show_scharr, show_laplacian, show_canny]):
                    tabs = []
                    
                    if show_sobel:
                        tabs.append("Sobel")
                    if show_scharr:
                        tabs.append("Scharr")
                    if show_laplacian:
                        tabs.append("Laplacian")
                    if show_canny:
                        tabs.append("Canny")
                    
                    selected_tab = st.tabs(tabs)
                    
                    for i, tab_name in enumerate(tabs):
                        with selected_tab[i]:
                            st.markdown(get_performance_notes(tab_name))
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try adjusting your parameters or uploading a different image.")

if __name__ == "__main__":
    run_app()