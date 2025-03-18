import json
import os

# COCO formatındaki json dosyasını ve çıktının kaydedileceği klasörü belirleyin
coco_json_path = "Dataset/helmetv2/test/_annotations.coco.json"  # COCO formatındaki JSON dosyasının yolu
output_labels_dir = "Dataset/helmetv2/test/labels/"  # YOLO formatındaki etiketlerin kaydedileceği klasör

# Klasör varsa silinmez, yoksa oluşturulur
os.makedirs(output_labels_dir, exist_ok=True)

# JSON dosyasını yükleme
with open(coco_json_path, "r") as file:
    coco_data = json.load(file)

images = {image["id"]: image for image in coco_data["images"]}
categories = {category["id"]: category["name"] for category in coco_data["categories"]}
annotations = coco_data["annotations"]

# Her bir annotation'ı işleyin
for annotation in annotations:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    bbox = annotation["bbox"]  # [x_min, y_min, width, height]
    
    # İlgili görsel bilgisi
    image_info = images[image_id]
    img_width = image_info["width"]
    img_height = image_info["height"]
    file_name = image_info["file_name"]

    # YOLO formatına dönüştürme
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height

    # YOLO formatındaki satır
    yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    # Etiket dosyasını kaydet
    label_file_path = os.path.join(output_labels_dir, f"{os.path.splitext(file_name)[0]}.txt")
    with open(label_file_path, "a") as label_file:
        label_file.write(yolo_line)

print(f"YOLO formatına dönüşüm tamamlandı. Etiketler {output_labels_dir} klasörüne kaydedildi.")
