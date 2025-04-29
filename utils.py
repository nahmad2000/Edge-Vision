import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess image for edge detection
    
    Args:
        image (numpy.ndarray): Input image
        size (tuple): Optional size to resize image to (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image
    
    Logic:
    1. Resize image if size is specified
    2. Convert to grayscale if image is color
    3. Apply optional noise reduction
    """
    try:
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Resize image if size is specified
        if size and (processed.shape[1], processed.shape[0]) != size:
            processed = cv2.resize(processed, size, interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image to {size}")
        
        # Convert to grayscale if image is color
        if len(processed.shape) > 2 and processed.shape[2] >= 3:
            # Keep original for display purposes
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            logger.info("Converted image to grayscale for processing")
            
            # Optional: Apply slight Gaussian blur to reduce noise
            # gray = cv2.GaussianBlur(gray, (3, 3), 0)
            # logger.info("Applied Gaussian blur for noise reduction")
            
            return gray
        
        return processed
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        # Return empty image of specified size or original size
        if size:
            return np.zeros(size[::-1], dtype=np.uint8)
        elif image is not None:
            if len(image.shape) > 2:
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                return np.zeros_like(image)
        else:
            return np.zeros((300, 300), dtype=np.uint8)

def display_results(results: List[np.ndarray], titles: Optional[List[str]] = None, cmap: str = 'gray') -> plt.Figure:
    """
    Create a grid display of edge detection results
    
    Args:
        results (list): List of result images
        titles (list): List of titles for each result
        cmap (str): Colormap for displaying results
    
    Returns:
        matplotlib.figure.Figure: Figure with plotted results
    
    Logic:
    1. Calculate grid size based on number of results
    2. Create subplots for each result
    3. Display each image with its title
    4. Adjust layout for clean display
    """
    try:
        # Validate inputs
        if not results:
            logger.warning("No results to display")
            fig, ax = plt.subplots(1, 1)
            ax.text(0.5, 0.5, "No results to display", ha='center', va='center')
            ax.axis('off')
            return fig
        
        # Set default titles if not provided
        if titles is None or len(titles) != len(results):
            titles = [f"Result {i+1}" for i in range(len(results))]
        
        # Calculate grid size
        n = len(results)
        cols = min(3, n)  # Max 3 columns
        rows = (n + cols - 1) // cols
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        
        # Ensure axes is always a 2D array for indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = np.array(axes).reshape(rows, cols)
        
        # Fill grid with images
        for i, (result, title) in enumerate(zip(results, titles)):
            r, c = i // cols, i % cols
            ax = axes[r, c]
            
            # Handle different image types
            if len(result.shape) == 3 and result.shape[2] == 3:
                # Color image
                ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                ax.imshow(result, cmap=cmap)
            
            ax.set_title(title)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(results), rows * cols):
            r, c = i // cols, i % cols
            axes[r, c].axis('off')
            axes[r, c].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        # Return a simple error figure
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, f"Error displaying results: {str(e)}", ha='center', va='center')
        ax.axis('off')
        return fig

def get_performance_notes(method_name: str) -> str:
    """
    Get performance notes for each edge detection method
    
    Args:
        method_name (str): Name of the edge detection method
    
    Returns:
        str: Performance characteristics and use cases
    
    Logic:
    1. Match method name to predefined notes
    2. Return relevant information about detector strengths/weaknesses
    """
    # Dictionary of performance notes for each method
    notes = {
        'sobel': """
        Sobel Operator:
        - Good at detecting edges in the vertical and horizontal directions
        - Less sensitive to noise compared to simple gradient methods
        - Computation is relatively fast and efficient
        - Works well for images with clear, strong edges
        - Performance degrades with complex textures or subtle transitions
        """,
        
        'scharr': """
        Scharr Operator:
        - More accurate gradient calculation than Sobel
        - Better rotation invariance for edge detection
        - Good for detecting fine details and subtle edges
        - Slightly more computationally intensive than Sobel
        - Can be more sensitive to noise in certain scenarios
        """,
        
        'laplacian': """
        Laplacian Operator:
        - Detects edges by finding zero crossings after applying the Laplacian
        - Good at finding the exact location of edges
        - Can detect edges in all directions equally
        - Very sensitive to noise - often requires pre-smoothing
        - Double edges may appear in the output
        - Works well for finding fine details in well-conditioned images
        """,
        
        'canny': """
        Canny Edge Detector:
        - Considered optimal for many applications
        - Multi-stage algorithm with noise reduction, gradient calculation, 
          non-maximum suppression, and hysteresis thresholding
        - Less susceptible to noise than other methods
        - Produces thin, well-connected edge lines
        - More computationally intensive
        - Requires parameter tuning for optimal results
        - Best overall performance for most general-purpose applications
        """
    }
    
    # Convert method name to lowercase and remove any whitespace
    method = method_name.lower().strip()
    
    # Return notes for the specified method or a default message
    return notes.get(method, f"No performance notes available for {method_name}")

def validate_image_path(file_path: str) -> bool:
    """
    Validate if the provided path points to a valid image file
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        bool: True if the file is a valid image, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        # Try to read the image
        img = cv2.imread(file_path)
        if img is None:
            logger.warning(f"Could not read image file: {file_path}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating image path: {str(e)}")
        return False