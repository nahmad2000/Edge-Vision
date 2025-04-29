# Edge Vision: Visual Edge Detection Comparator

A visual tool that applies multiple edge detection algorithms (Sobel, Laplacian, Canny, Scharr) to images and displays them side-by-side for easy comparison. Edge Vision enables users to adjust algorithm parameters through a simple interface and see the effects in real-time, making it ideal for computer vision learning and algorithm evaluation.

![Edge Vision Demo](https://raw.githubusercontent.com/nahmad2000/edge-vision/main/images/demo.png)

## Features

- Compare four popular edge detection algorithms side-by-side
- Interactive parameter adjustment for each algorithm
- Real-time visualization of parameter effects
- Easy image upload functionality
- Performance notes for each detector
- Simple, intuitive Streamlit interface

## Edge Detection Algorithms

1. **Sobel Operator**: Calculates the gradient of image intensity at each pixel, emphasizing regions of high spatial frequency.
2. **Scharr Operator**: A more accurate gradient calculation than Sobel with better rotation invariance.
3. **Laplacian Operator**: Uses the Laplacian filter to find areas of rapid intensity change.
4. **Canny Edge Detector**: Multi-stage algorithm that detects edges while suppressing noise.

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nahmad2000/edge-vision.git
   cd edge-vision
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

4. Create an images directory and add a sample image:
   ```bash
   mkdir -p images
   # Add your sample.jpg to the images directory
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run main.py
   ```

2. Access the web interface in your browser (typically http://localhost:8501).

3. Use the sidebar to:
   - Select or upload an image
   - Choose which edge detectors to apply
   - Adjust parameters for each detector
   - Toggle performance notes display

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
edge-vision/
├── images/                  # Sample images for testing
│   └── sample.jpg           # Example image
├── main.py                  # Entry point and UI
├── edge_detectors.py        # Core detection algorithms
├── utils.py                 # Helper functions
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Tips for Best Results

1. **Start with default parameters** and adjust incrementally.
2. **Canny detector** generally produces cleaner edges but requires more tuning.
3. For **noisy images**, lower thresholds may produce excessive edges.
4. The **Sobel and Scharr operators** are direction-sensitive; try changing dx/dy values.
5. **Laplacian** works best on images with sharp transitions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

