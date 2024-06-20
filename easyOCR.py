from ultralytics import YOLO
import cv2
from utils import read_license_plate
import numpy as np
import easyocr
import pandas as pd

def rotate_right(img):
    angle = -30
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def rotate_left(img):
    angle = -30
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def preprocess_LP_img(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
  gray = cv2.equalizeHist(gray)

  #Appy Gaussian Filter to reduce noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  return blurred

def main():
  # load model and image
  LP_model = YOLO('./model/LP_model.pt')
  img_name = "AQUA2_52911_checkin_2020-10-25-8-52w2paLJIm01.jpg"
  dataset_label_path = "./dataset/test/new_test.csv"
  dataset_img_path = "./dataset/test/test/"
  # img_path = dataset_img_path + img_name
  img_path = '/Users/hungnguyen/TaiLieu/test/number_license_plate_recognition/1.jpg'


  #read label 
  df = pd.read_csv(dataset_label_path)
  label_label = df['label']
  # get label of where df['filename] == img_name
  label = label_label[df['filename'] == img_name].values[0]

  print("label", label)

  img_input = cv2.imread(img_path)
  img = cv2.resize(img_input, (640, 640), interpolation=cv2.INTER_LINEAR)

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
  class_id = 'License Plate'
  conf = box.conf[0].item()
  

  # crop image to get license plate
  license_plate_before = img[(cords[1]):(cords[3]), (cords[0]):(cords[2])]
  cv2.imshow("images", license_plate_before)
  cv2.waitKey(0)
  x = 5
  license_plate = img[(cords[1]):(cords[3]), (cords[0]+x):(cords[2]-x)]
  # cv2.imshow("image", license_plate)
  # cv2.waitKey(0)

  # # preprocess license plate
  preprocessed_img = preprocess_LP_img(license_plate)

  reader = easyocr.Reader(['en'], gpu=False)
  detections = reader.readtext(preprocessed_img)
  detections = sorted(detections, key=lambda x: (x[0][1], x[0][0]))

  # read license plate
  license_plate_text, license_plate_text_score = read_license_plate(detections)

  # display image resized with bounding box license plate detection
  print(f"license_plate_text: {license_plate_text}")
  print(f"license_plate_text_score: {license_plate_text_score}")



if __name__ == '__main__':
  main()