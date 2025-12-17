# scripts/xmt_to_yolo.py
import xml.etree.ElementTree as ET
import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_DIR = os.path.join(BASE_DIR, "..", "dataset", "raw", "Indian_Number_Plates")
ANN_DIR = os.path.join(BASE_DIR, "..", "dataset", "raw", "Annotations")
OUT_DIR = os.path.join(BASE_DIR, "..", "dataset", "yolo")

print("IMG_DIR:", IMG_DIR)
print("ANN_DIR:", ANN_DIR)


os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

xml_files = os.listdir(ANN_DIR)
train, val = train_test_split(xml_files, test_size=0.2, random_state=42)

def convert(xml_file, split):
    tree = ET.parse(os.path.join(ANN_DIR, xml_file))
    root = tree.getroot()

    img_name = root.find("filename").text
    img_path = os.path.join(IMG_DIR, img_name)

    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)

    label_path = f"{OUT_DIR}/labels/{split}/{img_name[:-4]}.txt"

    with open(label_path, "w") as f:
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))


            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

    shutil.copy(img_path, f"{OUT_DIR}/images/{split}/{img_name}")

for x in train:
    convert(x, "train")

for x in val:
    convert(x, "val")

print("✅ Conversion Done")
