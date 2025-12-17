# inference/detect_plate.py
from ultralytics import YOLO
import cv2

model = YOLO("models/detector/best.pt")

def detect_plate(img_path):
    img = cv2.imread(img_path)
    results = model(img, conf=0.12, imgsz=640)
    
    plates = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = img[y1:y2, x1:x2]
        plates.append(plate)
        
    return plates
