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

def morphological_thinning(image):
    """
    Apply Zhang-Suen thinning algorithm to binary image.
    Args:
        image: Binary image as numpy array with values 0 and 255
    Returns:
        Thinned binary image
    """
    # Convert to binary image with 0 and 1
    image = image.copy() // 255
    
    def neighbors(x, y, image):
        """Return 8-neighbors of point p1(x,y) in clockwise order"""
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        return [image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1],     # P2,P3,P4,P5
                image[x1][y], image[x1][y_1], image[x][y_1], image[x_1][y_1]]   # P6,P7,P8,P9

    def transitions(neighbors):
        """Return number of 0,1 transitions in ordered sequence of neighbors"""
        n = neighbors + neighbors[0:1]    # P2, P3, ... P9, P2
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

    def first_subiteration(image):
        """Delete pixels satisfying all conditions in first subiteration"""
        rows, cols = image.shape
        deletion_candidates = []
        
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if image[x][y] != 1:
                    continue
                    
                P = neighbors(x, y, image)
                if (2 <= sum(P) <= 6 and           # Condition 1
                    transitions(P) == 1 and        # Condition 2
                    P[0] * P[2] * P[4] == 0 and   # Condition 3
                    P[2] * P[4] * P[6] == 0):     # Condition 4
                    deletion_candidates.append((x, y))
        
        for x, y in deletion_candidates:
            image[x][y] = 0
        return image

    def second_subiteration(image):
        """Delete pixels satisfying all conditions in second subiteration"""
        rows, cols = image.shape
        deletion_candidates = []
        
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if image[x][y] != 1:
                    continue
                    
                P = neighbors(x, y, image)
                if (2 <= sum(P) <= 6 and           # Condition 1
                    transitions(P) == 1 and        # Condition 2
                    P[0] * P[2] * P[6] == 0 and   # Condition 3
                    P[0] * P[4] * P[6] == 0):     # Condition 4
                    deletion_candidates.append((x, y))
        
        for x, y in deletion_candidates:
            image[x][y] = 0
        return image

    prev = np.zeros(image.shape, dtype=np.uint8)
    while True:
        image = first_subiteration(image)
        image = second_subiteration(image)
        if np.array_equal(image, prev):
            break
        prev = image.copy()
    
    # Convert back to 0-255 binary image
    return image * 255

def measure_cell(cell_mask):
    """Measures the cell length (3.06 pixels/micrometer)"""
    skeleton = morphological_thinning(cell_mask)
    length = np.count_nonzero(skeleton) / 3.06
    return skeleton, length