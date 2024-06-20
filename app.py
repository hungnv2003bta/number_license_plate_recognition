from ultralytics import YOLO
import cv2
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from utils import read_license_plate
import easyocr

# Set device to CPU
device = 'cpu'


# Load models
try:
    license_plate_detector = YOLO('./model/LP_model.pt').to(device)
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")

logging.basicConfig(level=logging.INFO)

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def preprocess_LP_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

@app.get("/")
async def read_root():
    return FileResponse("./static/index.html")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logging.error("Failed to decode image.")
            raise HTTPException(status_code=400, detail="Invalid image format.")
        logging.info("Image successfully read and decoded.")

        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Detect license plate
        results = license_plate_detector.predict(image)
        result = results[0]

        if len(result.boxes) == 0:
            raise HTTPException(status_code=404, detail="No license plate detected.")

        box = result.boxes[0]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = box.conf[0].item()

        # license_plate = image[cords[1]:cords[3], cords[0]:cords[2]]
        x = 5
        license_plate_cropped = image[(cords[1]):(cords[3]), (cords[0]+x):(cords[2]-x)]
        preprocessed_img = preprocess_LP_img(license_plate_cropped)
        reader = easyocr.Reader(['en'], gpu=False)
        text = reader.readtext(preprocessed_img)

        license_plate_text, license_plate_text_score = read_license_plate(text)

        response = {
            "license_plate": {
                "text": license_plate_text,
                "bbox_score": conf,
                "text_score": license_plate_text_score
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
