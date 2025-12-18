# 🚗 Automatic License Plate Recognition (ALPR)

An end-to-end **Automatic License Plate Recognition (ALPR)** system using **YOLOv8** for license plate detection and **OCR** for text recognition. This project covers dataset preparation, model training, inference, and result storage, and is structured for easy understanding and extension.

---

## 📌 Features

* License plate detection using **YOLOv8**
* OCR-based license number recognition
* Designed for **Indian vehicle number plates**
* Clean and modular project structure
* Easy inference on new images
* Ready for real-world applications

---

## 🧠 Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Object Detection:** YOLOv8 (Ultralytics)
* **Computer Vision:** OpenCV
* **OCR:** EasyOCR

---

## 📂 Project Structure

```
ALPR/
│
│
├── inference/
│   ├── detect_plate.py      # License plate detection logic
│   ├── recognize_text.py   # OCR on detected plate
│   └── main.py              # End-to-end inference pipeline
│
├── inference_results/       # Detection & OCR outputs
│
├── models/
│   └── detector/
│       └── best.pt          # Trained YOLOv8 model
│
├── runs/                    # YOLO training logs
│
├── scripts/
│   ├── train_detector.py   # YOLOv8 training script
│   └── xml_to_yolo.py      # XML → YOLO annotation converter
│
├── test_images/             # Images for testing
│
├── app.py                   # Entry point / demo script
├── yolov8n.pt               # Pretrained YOLOv8 model
├── temp_upload.jpg          # Temporary test image
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Vehicle images with annotated license plates
* Annotations converted from **XML to YOLO format**
* Dataset split into **training and validation** sets

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kinjal2103/ALPR.git
cd ALPR
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv alpr_env
alpr_env\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏗️ Dataset Preparation

Convert XML annotations to YOLO format:

```bash
python scripts/xml_to_yolo.py
```

This script:

* Reads XML annotations
* Converts bounding boxes to YOLO format
* Organizes images and labels properly

---

## 🏋️ Model Training

Train the YOLOv8 license plate detector:

```bash
python scripts/train_detector.py
```

* Base model: `yolov8n.pt`
* Best weights saved as:

```
models/detector/best.pt
```

---

## 🔍 Inference (Detection + OCR)

Run full ALPR pipeline on a test image:

```bash
python inference/main.py
```

What happens:

1. License plate is detected
2. Plate region is cropped
3. OCR extracts the license number
4. Results are saved in `inference_results/`

---

## 📈 Results

* Accurate detection for clear vehicle images
* Robust OCR for standard Indian number plates
* Output images and recognized text stored automatically

---

## 🚀 Future Improvements

* Real-time video and CCTV stream support
* License plate tracking
* Web-based interface
* Database integration
* Multi-country plate support

---

## 🧑‍💻 Author

**Kinjal Agarwal**
B.Tech CSE, IIT Patna

* GitHub: [https://github.com/Kinjal2103](https://github.com/Kinjal2103)

