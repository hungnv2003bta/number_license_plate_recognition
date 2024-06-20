from ultralytics import YOLO
import cv2
from utils import read_license_plate
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image

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
  
  return gray

def main():
  pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

  print(pytesseract.image_to_string(Image.open('/Users/hungnguyen/TaiLieu/FinalNhanDang/img/bien1.png')))


  # # load model and image
  # img_name = "AEONTP_62A02636_checkin_2020-1-13-16-14gXI_gF3J5Z.jpg"
  # LP_model = YOLO('./model/LP_model.pt')
  # dataset_label_path = "./dataset/test/new_test.csv"
  # dataset_img_path = "./dataset/test/test/"
  # img_path = "/Users/hungnguyen/TaiLieu/FinalNhanDang/img/img2.jpg"

  # #read label 
  # df = pd.read_csv(dataset_label_path)
  # label_label = df['label']
  # # get label of where df['filename] == img_name
  # label = label_label[df['filename'] == img_name].values[0]

  # print("label", label)

  # img = cv2.imread(img_path)
  # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
  
  # # get license plate
  # results = LP_model.predict(img)
  # result = results[0]
  # # check if can not detect license plate
  # if len(result.boxes) == 0:
  #   img = rotate_right(img)
  #   results = LP_model.predict(img)
  #   result = results[0]
  
  # if len(result.boxes) == 0:
  #   img = rotate_left(img)
  #   results = LP_model.predict(img)
  #   result = results[0]
  
  # box = result.boxes[0]

  # cords = box.xyxy[0].tolist()
  # cords = [round(x) for x in cords]
  # class_id = 'License Plate'
  # conf = box.conf[0].item()
  
  # # crop license plate
  # license_plate = img[cords[1]:cords[3], cords[0]:cords[2]]

  # gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

  # binary = cv2.threshold(gray, 68, 155, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  # text = pytesseract.image_to_string(binary, lang="eng", config='--psm 7')
  # print(text)

if __name__ == '__main__':
  main()