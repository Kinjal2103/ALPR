# scripts/train_detector.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/yolo/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16
)
