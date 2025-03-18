import os
from PIL import Image

def resize_images(source_path, target_path, size=(640, 640)):
    """
    Görselleri yeniden boyutlandırır ve hedef klasöre kaydeder.

    Args:
        source_path (str): Orijinal görsellerin bulunduğu klasör.
        target_path (str): Yeniden boyutlandırılmış görsellerin kaydedileceği klasör.
        size (tuple): Yeniden boyutlandırılacak boyut (varsayılan: (224, 224)).
    """
    # Hedef klasörü oluştur
    os.makedirs(target_path, exist_ok=True)

    # Görselleri yeniden boyutlandır ve hedef klasöre kaydet
    for file_name in os.listdir(source_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Sadece görsel dosyaları işleyin
            source_file = os.path.join(source_path, file_name)
            target_file = os.path.join(target_path, file_name)
            try:
                with Image.open(source_file) as img:
                    resized_img = img.resize(size)
                    resized_img.save(target_file)
                    print(f"Resized and saved: {target_file}")
            except Exception as e:
                print(f"Error resizing {source_file}: {e}")

# Kullanım
source_path = "helmet_Dataset_Demo/valid/images"  # Orijinal görsellerin bulunduğu klasörün yolu
target_path = "helmet_Dataset_Demo/valid/Resized_640"  # Yeniden boyutlandırılmış görsellerin kaydedileceği klasörün yolu
resize_images(source_path, target_path)