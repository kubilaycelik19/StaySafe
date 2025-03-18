import json

def merge_coco_jsons(json_file1, json_file2, output_file):
    # JSON dosyalarını yükle
    with open(json_file1, 'r') as f:
        coco1 = json.load(f)
    with open(json_file2, 'r') as f:
        coco2 = json.load(f)
    
    # Yeni JSON yapısı
    merged_coco = {"images": [], "annotations": [], "categories": []}

    # Benzersiz ID'ler için başlangıç değerleri
    annotation_id_offset = max([int(ann["id"]) for ann in coco1["annotations"]], default=0) + 1
    image_id_offset = len(coco1["images"])  # image_id string olduğu için sayısal bir düzenleme yapılmaz

    # Kategorileri birleştir (aynı kategoriler zaten mevcutsa, çakışmalar önlenir)
    categories = {cat["name"]: cat for cat in coco1["categories"]}
    for cat in coco2["categories"]:
        if cat["name"] not in categories:
            categories[cat["name"]] = {
                "id": len(categories) + 1,
                "name": cat["name"]
            }
    merged_coco["categories"] = list(categories.values())

    # Kategori adlarını ID'lere eşleştir
    category_name_to_id = {cat["name"]: cat["id"] for cat in merged_coco["categories"]}

    # Görüntüleri birleştir
    for img in coco1["images"]:
        merged_coco["images"].append(img)
    for img in coco2["images"]:
        new_img = img.copy()
        new_img["id"] = str(int(img["id"]) + image_id_offset)  # Benzersiz ID için düzenleme
        merged_coco["images"].append(new_img)

    # Anotasyonları birleştir
    for ann in coco1["annotations"]:
        merged_coco["annotations"].append(ann)
    for ann in coco2["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] = int(ann["id"]) + annotation_id_offset  # Benzersiz ID için düzenleme
        new_ann["image_id"] = str(int(ann["image_id"]) + image_id_offset)  # image_id düzenlemesi
        new_ann["category_id"] = category_name_to_id[next(cat["name"] for cat in coco2["categories"] if cat["id"] == ann["category_id"])]
        merged_coco["annotations"].append(new_ann)

    # JSON'u kaydet
    with open(output_file, "w") as f:
        json.dump(merged_coco, f, indent=4)

# Kullanım
json_file1 = "output.json"
json_file2 = "output2.json"
output_file = "merged.json"
merge_coco_jsons(json_file1, json_file2, output_file)