import streamlit as st
import cv2
import numpy as np
from PIL import Image
from inference.detect_plate import detect_plate
from inference.recognize_text import recognize_text

st.set_page_config(page_title="Indian ALPR System", layout="centered")

st.title("🚗 Indian License Plate Recognition")
st.write("Upload a car image to detect and read the number plate.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Run Detection'):
        with st.spinner('Processing...'):
            # Use your existing logic
            # Note: You might need to adjust detect_plate to accept an image array 
            # instead of just a path, or save the upload temporarily.
            temp_path = "temp_upload.jpg"
            cv2.imwrite(temp_path, img_cv)
            
            plates = detect_plate(temp_path)
            
            if len(plates) > 0:
                st.success(f"Found {len(plates)} plate(s)!")
                cols = st.columns(len(plates))
                
                for i, plate in enumerate(plates):
                    text = recognize_text(plate)
                    
                    with cols[i]:
                        # Convert plate to RGB for Streamlit display
                        plate_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
                        st.image(plate_rgb, caption=f"Crop {i+1}")
                        st.markdown(f"**Detected Text:** `{text}`")
            else:
                st.warning("No license plates detected.")