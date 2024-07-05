from ultralytics import YOLO
import cv2
from utils import read_license_plate
import numpy as np
import easyocr
import pandas as pd
import os
import math

def rotate_right(img):
    angle = -30
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def rotate_left(img):
    angle = -60 
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def find_horizontal_lines(lines):
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 15:  # Kiểm tra xem đường thẳng có ngang không
                horizontal_lines.append(line[0])
    return horizontal_lines

def find_nearest_horizontal_lines(horizontal_lines, image_height):
    center_y = image_height // 2
    min = image_height * 0.15
    top_line = None
    bottom_line = None
    min_top_dist = float('inf')
    min_bottom_dist = float('inf')

    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        dist_to_center = abs(y1 - center_y)
        if dist_to_center > min:
            if y1 < center_y:
                if dist_to_center < min_top_dist:
                    min_top_dist = dist_to_center
                    top_line = line
            else:
                if dist_to_center < min_bottom_dist:
                    min_bottom_dist = dist_to_center
                    bottom_line = line

    return top_line, bottom_line

def preprocess_LP_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    #Appy Gaussian Filter to reduce noise
    img_blurred = cv2.GaussianBlur(gray, (3,3), 0)

    edges = cv2.Canny(img_blurred,50, 170)

    length = img.shape[1] * 0.5

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=100, minLineLength=length, maxLineGap=10)

    if lines is not None: 
        max_length = 0
        best_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > max_length:
                if abs(y2- y1) / abs(x2 - x1) < 1.7:
                    best_line = line
                    max_length = length
        if best_line is not None:    
            x1, y1, x2, y2 = best_line[0]
            angle = calculate_angle(x1, y1, x2, y2)
            img_blurred = rotate(img_blurred, angle)

    edges = cv2.Canny(img_blurred, 50, 150)

    length = img.shape[1] * 0.5

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=length, maxLineGap=10)

    horizontal_lines = find_horizontal_lines(lines)

    image_height = img_blurred.shape[0]
    top_line, bottom_line = find_nearest_horizontal_lines(horizontal_lines, image_height)

    top_y = top_line[1] if top_line is not None else 0
    bottom_y = bottom_line[3] if bottom_line is not None else image_height

    # Cắt ảnh theo biên trên và dưới tìm được
    img_blurred = cv2.equalizeHist(img_blurred)
    cropped_image = img_blurred[top_y:bottom_y, 10:-5]

    thresh = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)

    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    total_pixels = thresh.shape[0] * thresh.shape[1]
    lower = total_pixels // 80
    upper = total_pixels // 15

    for (i, label) in enumerate(np.unique(labels)):
    # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,

        if numPixels > lower and numPixels < upper:
            left_edge_pixels = labelMask[:,0]
            right_edge_pixels = labelMask[:,-1]
            if np.any(left_edge_pixels == 255) or np.any(right_edge_pixels == 255):
                continue
            mask = cv2.add(mask, labelMask)

    return mask


def main():
    LP_model = YOLO('./model/LP_model.pt')
    dataset_label_path = "./dataset/test/new_test.csv"
    image_dataset = "./dataset/test/test/"
    image_name = "AQUA4_6767_checkin_2020-10-31-17-23M_8qb1K8Ow.jpg"
    img_path = image_dataset + image_name

    df = pd.read_csv(dataset_label_path)
    label_label = df['label']
    label = label_label[df['filename'] == image_name].values[0]


    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # get license plate
    results = LP_model.predict(img)
    result = results[0]
    
    # check if can not detect license plate
    if len(result.boxes) == 0:
        img = rotate_right(img)
        results = LP_model.predict(img)
        result = results[0]
    
    if len(result.boxes) == 0:
        img = rotate_left(img)
        results = LP_model.predict(img)
        result = results[0]
    
    box = result.boxes[0]

    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]

    # crop license plate
    license_plate = img[cords[1]:cords[3], cords[0]:cords[2]]

    preprocessed_img = preprocess_LP_img(license_plate)
    # cv2.imshow("Preprocessed image", preprocessed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    license_plate_text, _ = read_license_plate(preprocessed_img)

    print("Predicted license plate: ", license_plate_text)

if __name__ == "__main__":
    main()