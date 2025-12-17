
# inference/main.py
from detect_plate import detect_plate
from recognize_text import recognize_text
from ultralytics import YOLO
import cv2
import os

# Load the model here to get coordinates
model = YOLO("models/detector/best.pt")
img_path = "test_images/car1.jpg"
img = cv2.imread(img_path)
display_img = img.copy()

# Run detection
results = model(img)

for result in results:
    for box in result.boxes:
        # 1. Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        
        # 2. Crop for OCR
        plate_crop = img[y1:y2, x1:x2]
        plate_text = recognize_text(plate_crop)
        
        # 3. Draw Bounding Box (Green)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 4. Draw Label Background and Text
        label = f"{plate_text} ({conf:.2f})"
        cv2.putText(display_img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Detected: {plate_text}")

# Save and Show result
cv2.imwrite("inference_results/detected_car.jpg", display_img)
cv2.imshow("ALPR Result", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()