from ultralytics import YOLO
import cv2
from utils import read_license_plate
import numpy as np
import easyocr
import pandas as pd
import os

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
  dataset_img_path = "./dataset/test/test"
  image_name = os.listdir(dataset_img_path)
  #read label
  df = pd.read_csv(dataset_label_path)
  label_label = df['label']
  
  cnt = 0
  err = []
  img_err = []
  
  for idx, img_name in enumerate(image_name):
    img_path = os.path.join(dataset_img_path, img_name)

    # get label of where df['filename] == img_name
    matching_rows = df[df['filename'] == img_name]

    if not matching_rows.empty:
    # Nếu có hàng khớp, lấy giá trị của cột 'label'
      label = matching_rows['label'].values[0]
    else:
    # Nếu không có hàng nào khớp, in ra img_name
      print(img_name)
      continue
    
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
    class_id = 'License Plate' + str(idx)
    conf = box.conf[0].item()
    
    print(class_id)
    # crop license plate
    license_plate = img[cords[1]:cords[3], cords[0]:cords[2]]

    preprocessed_img = preprocess_LP_img(license_plate)

    # save_path = os.path.join('./img', f"{class_id}.jpg")
    # cv2.imwrite(save_path, license_plate)

    license_plate_text, license_plate_text_score = read_license_plate(preprocessed_img)

    # print(f"license_plate_text: {license_plate_text}")
    # print(f"license_plate_text_score: {license_plate_text_score}")
    if license_plate_text == label:
      cnt += 1
    else:
      err.append(license_plate_text)
      img_err.append(class_id)
      
  print(f"Accuracy: {cnt} / {len(df)}")
  for idx in range(len(err)): 
    print(err[idx],' ', img_err[idx])

if __name__ == '__main__':
  main()