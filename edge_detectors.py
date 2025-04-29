import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_sobel(image, ksize=3, scale=1, delta=0, dx=1, dy=1):
    """
    Apply Sobel edge detection to an image
    
    Args:
        image (numpy.ndarray): Input image
        ksize (int): Kernel size (must be 1, 3, 5, or 7)
        scale (float): Scale factor for derivatives
        delta (float): Value added to results
        dx (int): Order of derivative x
        dy (int): Order of derivative y
    
    Returns:
        numpy.ndarray: Edge detection result
    
    Logic:
    1. Convert image to grayscale if needed
    2. Apply Sobel operator in x and y directions
    3. Combine x and y gradients
    4. Normalize result for display
    """
    try:
        # Validate kernel size
        if ksize not in (1, 3, 5, 7):
            logger.warning(f"Invalid kernel size {ksize} for Sobel. Using default size 3.")
            ksize = 3
        
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Sobel operator in x and y directions
        grad_x = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize, scale=scale, delta=delta)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize, scale=scale, delta=delta)
        
        # Combine gradients
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Normalize for display
        result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in Sobel edge detection: {str(e)}")
        return np.zeros_like(image) if len(image.shape) <= 2 else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

def apply_scharr(image, scale=1, delta=0, dx=1, dy=0):
    """
    Apply Scharr edge detection to an image
    
    Args:
        image (numpy.ndarray): Input image
        scale (float): Scale factor for derivatives
        delta (float): Value added to results
        dx (int): Order of derivative x
        dy (int): Order of derivative y
    
    Returns:
        numpy.ndarray: Edge detection result
    
    Logic:
    1. Convert image to grayscale if needed
    2. Apply Scharr operator in x and y directions
    3. Combine x and y gradients
    4. Normalize result for display
    """
    try:
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Scharr operator in x and y directions
        if dx == 1 and dy == 0:
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
        elif dx == 0 and dy == 1:
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
        else:
            # Default to both x and y derivatives
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
        
        # Combine gradients
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Normalize for display
        result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in Scharr edge detection: {str(e)}")
        return np.zeros_like(image) if len(image.shape) <= 2 else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

def apply_laplacian(image, ksize=3, scale=1, delta=0):
    """
    Apply Laplacian edge detection to an image
    
    Args:
        image (numpy.ndarray): Input image
        ksize (int): Kernel size
        scale (float): Scale factor for the Laplacian
        delta (float): Value added to results
    
    Returns:
        numpy.ndarray: Edge detection result
    
    Logic:
    1. Convert image to grayscale if needed
    2. Apply Laplacian operator
    3. Normalize result for display
    """
    try:
        # Validate kernel size
        if ksize not in (1, 3, 5, 7):
            logger.warning(f"Invalid kernel size {ksize} for Laplacian. Using default size 3.")
            ksize = 3
        
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise (optional)
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
        
        # Take absolute value and convert to 8-bit
        abs_laplacian = np.absolute(laplacian)
        result = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in Laplacian edge detection: {str(e)}")
        return np.zeros_like(image) if len(image.shape) <= 2 else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

def apply_canny(image, threshold1=100, threshold2=200, aperture_size=3, L2gradient=False):
    """
    Apply Canny edge detection to an image
    
    Args:
        image (numpy.ndarray): Input image
        threshold1 (float): First threshold for hysteresis procedure
        threshold2 (float): Second threshold for hysteresis procedure
        aperture_size (int): Aperture size for the Sobel operator
        L2gradient (bool): Flag to use L2 norm for gradient magnitude
    
    Returns:
        numpy.ndarray: Edge detection result
    
    Logic:
    1. Convert image to grayscale if needed
    2. Apply Gaussian blur to reduce noise
    3. Apply Canny algorithm
    4. Return binary edge map
    """
    try:
        # Validate aperture size
        if aperture_size not in (3, 5, 7):
            logger.warning(f"Invalid aperture size {aperture_size} for Canny. Using default size 3.")
            aperture_size = 3
        
        # Ensure thresholds are in correct order
        if threshold1 > threshold2:
            threshold1, threshold2 = threshold2, threshold1
            logger.warning("Threshold1 was greater than threshold2. Values have been swapped.")
        
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny algorithm
        edges = cv2.Canny(blurred, threshold1, threshold2, apertureSize=aperture_size, L2gradient=L2gradient)
        
        return edges
    
    except Exception as e:
        logger.error(f"Error in Canny edge detection: {str(e)}")
        return np.zeros_like(image) if len(image.shape) <= 2 else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

def compare_all(image, params=None):
    """
    Apply all edge detection methods to an image with specified parameters
    
    Args:
        image (numpy.ndarray): Input image
        params (dict): Parameters for each detector (optional)
    
    Returns:
        dict: Dictionary with results from all edge detectors
    
    Logic:
    1. Set default parameters if not provided
    2. Apply each edge detection method
    3. Return dictionary with labeled results
    """
    try:
        # Initialize default parameters
        default_params = {
            'sobel': {'ksize': 3, 'scale': 1, 'delta': 0, 'dx': 1, 'dy': 1},
            'scharr': {'scale': 1, 'delta': 0, 'dx': 1, 'dy': 0},
            'laplacian': {'ksize': 3, 'scale': 1, 'delta': 0},
            'canny': {'threshold1': 100, 'threshold2': 200, 'aperture_size': 3, 'L2gradient': False}
        }
        
        # Use provided parameters or defaults
        detector_params = default_params
        if params:
            for method, method_params in params.items():
                if method in detector_params and method_params:
                    # Update only provided parameters
                    detector_params[method].update(method_params)
        
        # Apply each edge detection method
        results = {
            'original': image,
            'sobel': apply_sobel(image, **detector_params['sobel']),
            'scharr': apply_scharr(image, **detector_params['scharr']),
            'laplacian': apply_laplacian(image, **detector_params['laplacian']),
            'canny': apply_canny(image, **detector_params['canny'])
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error comparing edge detectors: {str(e)}")
        return {'original': image, 'error': str(e)}