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

def convert2Square(image):
    # Function to convert image to a square by padding
    h, w = image.shape[:2]
    if h > w:
        diff = h - w
        pad1, pad2 = diff // 2, diff - diff // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        diff = w - h
        pad1, pad2 = diff // 2, diff - diff // 2
        image = cv2.copyMakeBorder(image, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image

def preprocess_LP_img(img):
  img = convert2Square(img)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
  gray = cv2.equalizeHist(gray)

  #Appy Gaussian Filter to reduce noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  #---------------------------------
  blurred = cv2.bilateralFilter(blurred, 5, 50, 50)

  threshold = cv2.threshold(blurred,140, 255, cv2.THRESH_BINARY_INV)
  #---------------------------------
  return threshold


def main():
  # load model and image
  img_name = "AQUA2_09290_checkin_2020-10-22-10-47kCHb4hd9Qd.jpg"
  LP_model = YOLO('./model/LP_model.pt')
  dataset_label_path = "./dataset/test/new_test.csv"
  dataset_img_path = "./dataset/test/test/"
  img_path = dataset_img_path + img_name

  #read label 
  df = pd.read_csv(dataset_label_path)
  label_label = df['label']
  # get label of where df['filename] == img_name
  label = label_label[df['filename'] == img_name].values[0]

  print("label", label)

  img = cv2.imread(img_path)
  # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
  
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
  license_plate = img[cords[1]:cords[3], cords[0]:cords[2]]

  # preprocessed_img = preprocess_LP_img(license_plate)

  # license_plate_crop_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
  # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray,140, 255, cv2.THRESH_BINARY_INV)

  # cv2.imshow('preprocessed_img', license_plate_crop_thresh)
  # cv2.waitKey(0)

  preprocess_LP_img = preprocess_LP_img(license_plate)

  easy = easyocr.Reader(['en'])
  detections = easy.readtext(preprocess_LP_img )

  if(len(detections) == 2):
      bbox1, text1, score1 = detections[0]
      bbox2, text2, score2 = detections[1]
      score = (score1 + score2) / 2
      text = text1 + text2
      print(text)
  elif len(detections) == 1:
      bbox1, text1, score1 = detections[0]
      score = score1
      text = text1
      print(text)
  else:
      print("No text detected")


  # license_plate_text, license_plate_text_score = read_license_plate(preprocessed_img)

  # print(f"license_plate_text: {license_plate_text}")
  # print(f"license_plate_text_score: {license_plate_text_score}")



if __name__ == '__main__':
  main()