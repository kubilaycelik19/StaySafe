import os
import json
from pathlib import Path

# YOLO etiketleri ve görüntü bilgileri
yolo_labels_dir = "helmet_dataset/labels/test"  # YOLO formatındaki etiketlerin klasörü
images_dir = "helmet_dataset/images/test"  # Görüntülerin bulunduğu klasör
output_coco_file = "test_annotations.json"  # Çıkış JSON dosyası

# COCO formatında sabit alanlar
coco_format = {
    "info": {
        "description": "Converted YOLO to COCO",
        "version": "1.0",
        "year": 2024
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "baret"},
        {"id": 1, "name": "insan"}
    ]
}

# Helper: YOLO formatını COCO'ya çevir
def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]

# Görüntü ve etiketleri işle
annotation_id = 0
for label_file in os.listdir(yolo_labels_dir):
    if label_file.endswith(".txt"):
        # Görüntü bilgileri
        image_id = Path(label_file).stem  # Dosya adı, sayı yerine string olarak kullanılır
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue  # Görüntü eksikse atla
        img_width, img_height = 640, 640  # Görüntü boyutunu burada belirt

        # COCO görüntü bilgisi
        coco_format["images"].append({
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "width": img_width,
            "height": img_height
        })

        # Etiket bilgilerini oku
        with open(os.path.join(yolo_labels_dir, label_file), "r") as f:
            for line in f.readlines():
                yolo_data = list(map(float, line.strip().split()))
                class_id = int(yolo_data[0])
                bbox = yolo_to_coco_bbox(yolo_data[1:], img_width, img_height)

                # COCO anotasyonu
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,  # String olarak kullanılabilir
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

# JSON dosyasını kaydet
with open(output_coco_file, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO formatına dönüştürme tamamlandı: {output_coco_file}")
