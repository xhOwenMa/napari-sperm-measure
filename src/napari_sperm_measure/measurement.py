"""
Core measurement functionality module.
Contains functions for image preprocessing, cell tracing, and length measurement.
"""

import numpy as np
import cv2


def initial_preprocessing(image, iterations=5, kernel_size=11, block_size=51, c_value=-3):
    """Enhanced preprocessing with adaptive methods and configurable parameters"""
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Ensure image is uint8
    image = image.astype(np.uint8)
    
    # Normalize to full range
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    # Apply adaptive thresholding with configurable parameters
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,  # configurable
        C=c_value  # configurable
    )
    
    # close the gaps on the cell edges
    adaptive_thresh_copy = adaptive_thresh.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(iterations):
        closing = cv2.morphologyEx(adaptive_thresh_copy, cv2.MORPH_CLOSE, kernel)
        adaptive_thresh_copy = closing
    
    # Ensure all stages are uint8 and non-empty
    stages = {
        'Preprocessed Image': closing.astype(np.uint8)
    }

    return stages

def trace_cell(image, point):
    """Traces the cell body by flooding from the initial point"""
    # Ensure the image is binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Check if the seed point is on a cell (should be 255)
    if binary[point[0], point[1]] != 255:
        print("Seed point is not on a cell. Please click inside the cell.")
        return np.zeros_like(binary)

    # Make a copy of the binary image to perform flood fill
    flood_filled = binary.copy()

    # Create a mask (needs to be size h+2 by w+2)
    h, w = binary.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Use a unique value for flood fill
    new_val = 128

    # Perform flood fill from the seed point
    cv2.floodFill(flood_filled, mask, seedPoint=(point[1], point[0]), newVal=new_val)

    # Create the cell mask by extracting the filled area
    cell_mask = np.zeros_like(binary)
    cell_mask[flood_filled == new_val] = 255

    # Return the cell mask
    return cell_mask

def measure_cell(cell_mask):
    """Measures the cell length using morphological operations (3.06 pixels/micrometer)"""
    def morphological_thinning(image):
        """Apply enhanced morphological thinning to the binary image."""
        size = np.size(image)
        skel = np.zeros(image.shape, np.uint8)
        
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        
        while not done:
            eroded = cv2.erode(image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(image, temp)
            skel = cv2.bitwise_or(skel, temp)
            image = eroded.copy()
            
            zeros = size - cv2.countNonZero(image)
            if zeros == size:
                done = True
        
        return skel

    eroded = morphological_thinning(cell_mask)
    # 2, calculate the area of the cell in micrometers
    area = np.sum(eroded / 255) / 3.06
    return eroded, area