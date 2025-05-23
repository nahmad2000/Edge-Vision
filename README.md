# Edge Vision: Visual Edge Detection Comparator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://edge-vision-nahmad.streamlit.app/)

A visual tool that applies multiple edge detection algorithms (Sobel, Laplacian, Canny, Scharr) to images and displays them side-by-side for easy comparison. Edge Vision enables users to adjust algorithm parameters through a simple interface and see the effects in real-time, making it ideal for computer vision learning and algorithm evaluation.

![Edge Vision Demo 1](images/demo1.png) 
![Edge Vision Demo 2](images/demo2.png)

## Features

- Compare four popular edge detection algorithms side-by-side
- Interactive parameter adjustment for each algorithm
- Real-time visualization of parameter effects
- Easy image upload functionality
- Performance notes for each detector
- Simple, intuitive Streamlit interface

## Usage

### Live Demo (Recommended)

You can use the deployed application directly in your browser without any installation:

**➡️ [https://edge-vision-nahmad.streamlit.app/](https://edge-vision-nahmad.streamlit.app/)**

Simply open the link and use the sidebar to:
- Select or upload an image
- Choose which edge detectors to apply
- Adjust parameters for each detector
- Toggle performance notes display

### Local Usage

If you want to run the application on your own machine (e.g., for development or offline use):

1.  **(Prerequisite)** Follow the Installation steps below.
2.  Start the application from your terminal within the cloned project directory:
    ```bash
    streamlit run main.py
    ```
3.  Access the web interface in your browser (typically http://localhost:8501).

## Installation (for Local Usage Only)

Follow these steps only if you want to run the code on your own machine. If you just want to use the tool, use the Live Demo link above.

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nahmad2000/Edge-Vision.git
   cd Edge-Vision
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For deployment on Linux servers like Streamlit Cloud, system dependencies listed in `packages.txt` might also be needed).*

4. Ensure you have an images directory (if using the default image option):
   ```bash
   mkdir -p images
   # Add your sample image (e.g., sample.jpg or demo1.png) to the images directory
   # Make sure the path matches the DEFAULT_IMAGE variable in main.py
   ```

## Edge Detection Algorithms

1.  **Sobel Operator**: Calculates the gradient of image intensity at each pixel, emphasizing regions of high spatial frequency.
2.  **Scharr Operator**: A more accurate gradient calculation than Sobel with better rotation invariance.
3.  **Laplacian Operator**: Uses the Laplacian filter to find areas of rapid intensity change.
4.  **Canny Edge Detector**: Multi-stage algorithm that detects edges while suppressing noise.

## Parameter Explanations

### Sobel Parameters
- **Kernel Size**: Size of the Sobel kernel (1, 3, 5, or 7)
- **Scale**: Scale factor for the computed derivatives
- **Delta**: Value added to the results
- **X/Y Derivative Order**: Order of the derivative in x/y direction

### Scharr Parameters
- **Scale**: Scale factor for the computed derivatives
- **Delta**: Value added to the results
- **X/Y Direction**: Direction flags for gradient calculation

### Laplacian Parameters
- **Kernel Size**: Size of the Laplacian kernel
- **Scale**: Scale factor for the computed Laplacian
- **Delta**: Value added to the results

### Canny Parameters
- **Threshold 1**: First threshold for the hysteresis procedure
- **Threshold 2**: Second threshold for the hysteresis procedure
- **Aperture Size**: Aperture size for the Sobel operator used by Canny
- **L2 Gradient**: Flag to use L2 norm for gradient magnitude (more accurate but slower)

## Repository Structure

```
Edge-Vision/
├── images/              # Sample images for testing (ensure default image exists here)
│   └── ...
├── main.py              # Entry point and UI
├── edge_detectors.py    # Core detection algorithms
├── utils.py             # Helper functions
├── requirements.txt     # Python dependencies for pip
├── packages.txt         # System dependencies for apt-get (used by Streamlit Cloud)
└── README.md            # Project documentation
```

## Tips for Best Results

1.  **Start with default parameters** and adjust incrementally.
2.  **Canny detector** generally produces cleaner edges but requires more tuning.
3.  For **noisy images**, lower thresholds may produce excessive edges.
4.  The **Sobel and Scharr operators** are direction-sensitive; try changing dx/dy values.
5.  **Laplacian** works best on images with sharp transitions.
