import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class StaySafeDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        # Görüntüyü yükle
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Label'ı oku
        labels = np.loadtxt(label_path).reshape(-1, 5)  # [class_id, x_center, y_center, width, height]
        
        # YOLO formatından Albumentations formatına dönüştür
        boxes = []
        class_ids = []
        for label in labels:
            class_id, x_center, y_center, width, height = label
            x_min = (x_center - width/2) * img.shape[1]
            y_min = (y_center - height/2) * img.shape[0]
            x_max = (x_center + width/2) * img.shape[1]
            y_max = (y_center + height/2) * img.shape[0]
            boxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(class_id)

        # Transform uygula
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=class_ids)
            img = transformed['image']
            boxes = transformed['bboxes']
            class_ids = transformed['labels']

            # Albumentations formatından YOLO formatına geri dönüştür
            labels = []
            for box, class_id in zip(boxes, class_ids):
                x_min, y_min, x_max, y_max = box
                x_center = ((x_min + x_max) / 2) / img.shape[1]
                y_center = ((y_min + y_max) / 2) / img.shape[0]
                width = (x_max - x_min) / img.shape[1]
                height = (y_max - y_min) / img.shape[0]
                labels.append([class_id, x_center, y_center, width, height])

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        labels = torch.tensor(labels, dtype=torch.float32)

        return img, labels

# Albumentations transform pipeline
transform_albumentations = A.Compose([
    A.Resize(640, 640),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Dataset oluştur
train_dataset = StaySafeDataset(
    img_dir="C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/train/images",
    label_dir="C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/train/labels",
    transform=transform_albumentations
)

valid_dataset = StaySafeDataset(
    img_dir="C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/valid/images",
    label_dir="C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/valid/labels",
    transform=transform_albumentations
)

test_dataset = StaySafeDataset(
    img_dir="C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/test/images",
    label_dir="C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/test/labels",
    transform=transform_albumentations
)

# DataLoader oluştur
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
