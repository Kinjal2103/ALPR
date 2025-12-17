# inference/recognize_text.py

import easyocr
import cv2
import re

reader = easyocr.Reader(['en'])

def recognize_text(plate_img):
    # 1. Simple Resize and Contrast
    # EasyOCR works best when characters are at least 30-40 pixels tall
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Add Padding
    # OCR engines fail if text is too close to the image edge. 
    # This adds a white border around the plate.
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 3. OCR with Paragraph mode
    # Paragraph mode helps group split text blocks (like IND and the numbers)
    result = reader.readtext(gray, paragraph=True)

    if not result:
        return ""

    # Clean the output
    full_text = " ".join([res[1] for res in result])
    clean_text = full_text.replace("IND", "")
    return "".join(re.findall("[A-Z0-9]", clean_text))