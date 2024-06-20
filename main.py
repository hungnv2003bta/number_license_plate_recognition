from ultralytics import YOLO
import cv2
from utils import read_license_plate
import pandas as pd
import os
import numpy as np
import easyocr

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
  dataset_label_path = "./dataset/test/new_test.csv"
  dataset_img_path = "./dataset/test/test/"
  df = pd.read_csv(dataset_label_path)

  # load test set and do predict on each image, df include filename and label
  cnt = 0
  for index, row in df.iterrows():
      img_name = row['filename']
      label = row['label']
      img_path = dataset_img_path + img_name

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
      class_id = 'License Plate'
      conf = box.conf[0].item()
      
      # crop license plate
      # license_plate = img[cords[1]:cords[3], cords[0]:cords[2]]
      x = 5
      license_plate_cropped = img[(cords[1]):(cords[3]), (cords[0]+x):(cords[2]-x)]

      # preprocess license plate
      preprocessed_img = preprocess_LP_img(license_plate_cropped)

      reader = easyocr.Reader(['en'], gpu=False)
      detections = reader.readtext(preprocessed_img)
      detections = sorted(detections, key=lambda x: (x[0][1], x[0][0]))

      # read license plate
      license_plate_text, license_plate_text_score = read_license_plate(detections)

      # print(f"license_plate_text: {license_plate_text}")
      if license_plate_text == label:
        cnt += 1
      else:
        #  write to file ./model/wrong_predict.csv as filename, label, predict
        with open('./model/wrong_predict.csv', 'a') as f:
          f.write(f"{img_name},{label},{license_plate_text}\n")

  print(f"Accuracy: {cnt} / {len(df)}")
  



if __name__ == '__main__':
  main()