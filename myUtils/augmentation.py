import os
from PIL import Image, ImageOps, ImageEnhance

def augment_images(source_path, target_path, rotation_angles=[35, 70]):
    """
    Görüntülere augmentation uygular:
    - Saat yönünün tersine döndürme
    - İlk döndürmede aynalama, ikinci döndürmede kontrast artırma

    Args:
        source_path (str): Orijinal görsellerin bulunduğu klasör.
        target_path (str): Augmentation uygulanmış görsellerin kaydedileceği klasör.
        rotation_angles (list): Döndürme açıları (varsayılan: [35, 70]).
    """
    # Hedef klasörü oluştur
    os.makedirs(target_path, exist_ok=True)

    # Görselleri işle
    for file_name in os.listdir(source_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            source_file = os.path.join(source_path, file_name)
            try:
                with Image.open(source_file) as img:
                    for idx, angle in enumerate(rotation_angles):
                        rotated_img = img.rotate(-angle, expand=True)  # Saat yönünün tersine döndürme
                        
                        if idx == 0:  # İlk döndürme: Aynalama
                            mirrored = ImageOps.mirror(rotated_img)
                            mirrored.save(os.path.join(target_path, f"{file_name.split('.')[0]}_rot-{angle}_mirror.jpg"))
                            print(f"Saved mirrored: {file_name}_rot-{angle}_mirror.jpg")
                        
                        elif idx == 1:  # İkinci döndürme: Kontrast artırma
                            enhancer = ImageEnhance.Contrast(rotated_img)
                            contrast_img = enhancer.enhance(1.7)  # Kontrastı artır
                            contrast_img.save(os.path.join(target_path, f"{file_name.split('.')[0]}_rot-{angle}_contrast.jpg"))
                            print(f"Saved contrast adjusted: {file_name}_rot-{angle}_contrast.jpg")
                            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Kullanım
source_path = "helmet_Dataset_Demo/valid/Resized_640"  # Orijinal görsellerin bulunduğu klasörün yolu
target_path = "helmet_Dataset_Demo/valid/Resized_augmented_640"  # Augmentation uygulanmış görsellerin kaydedileceği klasörün yolu
augment_images(source_path, target_path)
