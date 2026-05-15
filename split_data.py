import os
import shutil
import random

# Create folders
os.makedirs("dataset/train/NORMAL", exist_ok=True)
os.makedirs("dataset/train/PNEUMONIA", exist_ok=True)
os.makedirs("dataset/val/NORMAL", exist_ok=True)
os.makedirs("dataset/val/PNEUMONIA", exist_ok=True)

source_normal = "NORMAL"
source_pneumonia = "PNEUMONIA"

def split_data(source, train, val, split_ratio=0.8):
    files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.png', '.jpeg'))]

    random.shuffle(files)
    split = int(len(files) * split_ratio)

    for f in files[:split]:
        shutil.copy(os.path.join(source, f), train)

    for f in files[split:]:
        shutil.copy(os.path.join(source, f), val)

split_data(source_normal, "dataset/train/NORMAL", "dataset/val/NORMAL")
split_data(source_pneumonia, "dataset/train/PNEUMONIA", "dataset/val/PNEUMONIA")

print("✅ Dataset ready!")