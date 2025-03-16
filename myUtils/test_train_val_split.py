import os
import shutil
import random

def create_directory_structure(base_dir):
    """Veri seti için gerekli klasör yapısını oluşturur"""
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    # Ana dizinleri oluştur
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    return train_dir

def move_data_to_train(source_dir, train_dir):
    """Kaynak dizindeki verileri train klasörüne taşır"""
    # Her bir kişi klasörü için
    for person_name in os.listdir(source_dir):
        src_person_path = os.path.join(source_dir, person_name)
        
        # Klasör değilse atla
        if not os.path.isdir(src_person_path):
            continue
            
        # Train dizininde kişi klasörünü oluştur
        dst_person_path = os.path.join(train_dir, person_name)
        
        # Eğer klasör zaten varsa, içeriğini sil
        if os.path.exists(dst_person_path):
            shutil.rmtree(dst_person_path)
            
        # Klasörü kopyala
        shutil.copytree(src_person_path, dst_person_path)
        print(f"{person_name} klasörü train dizinine kopyalandı.")

def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Veri setini train, validation ve test olarak böler.
    
    Args:
        base_dir (str): Ana veri seti dizini
        train_ratio (float): Eğitim seti oranı (0-1 arası)
        val_ratio (float): Validasyon seti oranı (0-1 arası)
        test_ratio (float): Test seti oranı (0-1 arası)
    """
    # Oranların toplamının 1 olduğunu kontrol et
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Oranların toplamı 1 olmalıdır!"
    
    # Ana dizini belirle
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    # Her sınıf klasörünü işle
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        
        # Eğer klasör değilse atla
        if not os.path.isdir(class_path):
            continue
        
        # Validasyon ve test klasörlerini oluştur
        val_class_path = os.path.join(val_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        # Tüm görüntüleri al
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)
        
        if total_images == 0:
            print(f"Uyarı: {class_name} klasöründe görüntü bulunamadı!")
            continue
        
        # Görüntüleri karıştır
        random.shuffle(images)
        
        # Bölme noktalarını hesapla
        val_split = int(total_images * (train_ratio))
        test_split = int(total_images * (train_ratio + val_ratio))
        
        # Görüntüleri böl
        val_images = images[val_split:test_split]
        test_images = images[test_split:]
        
        print(f"\n{class_name} sınıfı için bölümleme:")
        print(f"Toplam görüntü: {total_images}")
        print(f"Eğitim seti: {val_split} görüntü")
        print(f"Validasyon seti: {len(val_images)} görüntü")
        print(f"Test seti: {len(test_images)} görüntü")
        
        # Validasyon görüntülerini taşı
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(val_class_path, img)
            shutil.move(src_path, dst_path)
        
        # Test görüntülerini taşı
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(test_class_path, img)
            shutil.move(src_path, dst_path)

if __name__ == "__main__":
    # Veri seti dizinini belirle
    source_dir = "C:/Users/celik/Desktop/StaySafe/dataset/faces"
    
    # Dizinin var olduğunu kontrol et
    if not os.path.exists(source_dir):
        print(f"Hata: Kaynak dizin ({source_dir}) bulunamadı!")
        exit(1)
    
    # Hedef dizini belirle (bir üst dizinde "dataset" klasörü)
    base_dir = os.path.dirname(os.path.dirname(source_dir))  # İki üst dizine çık
    base_dir = os.path.join(base_dir, "dataset")  # dataset klasörünü hedefle
    os.makedirs(base_dir, exist_ok=True)
    print(f"\nHedef dizin: {base_dir}")
    
    # Train dizinini oluştur ve verileri kopyala
    train_dir = create_directory_structure(base_dir)
    print(f"\nKaynak dizin: {source_dir}")
    print("Veriler kopyalanıyor...")
    move_data_to_train(source_dir, train_dir)
    
    # Veri setini böl
    print("\nVeri seti bölümleme başlıyor...")
    split_dataset(
        base_dir=base_dir,
        train_ratio=0.7,  # %70 eğitim
        val_ratio=0.15,   # %15 validasyon
        test_ratio=0.15   # %15 test
    )
    
    print("\nVeri seti bölümleme tamamlandı!")
    print(f"Veriler {base_dir} dizininde train, val ve test olarak bölündü.")
