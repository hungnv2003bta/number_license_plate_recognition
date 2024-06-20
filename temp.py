import cv2
import numpy as np

# Step 1: Convert to Grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply a Two-Dimensional Filter (Bilateral Filter)
def apply_bilateral_filter(gray_image):
    return cv2.bilateralFilter(gray_image, 9, 75, 75)

# Step 3: Canny's Edge Detection
def canny_edge_detection(filtered_image):
    # Step 3a: Apply Gaussian Filter
    blurred_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
    
    # Step 3b-e: Canny Edge Detection
    edges = cv2.Canny(blurred_image, 50, 150)
    return edges

# Step 4: Trace the Image (Contour Detection)
def trace_image(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Load the image
image = cv2.imread('license_plate.jpg')

# Process the image
gray_image = convert_to_grayscale(image)
filtered_image = apply_bilateral_filter(gray_image)
edges = canny_edge_detection(filtered_image)
contours = trace_image(edges)

# Draw contours on the original image
output_image = image.copy()
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Segmented Characters', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()