import os
import sys # Python versiyon kontrolü için
import numpy as np
import json
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision.models import Resn

# FaceNet için import
try:
    from torchvision.models import InceptionResnetV1
except ImportError:
    print("HATA: facenet-pytorch kütüphanesi bulunamadı.")
    print("Lütfen 'pip install facenet-pytorch' komutu ile yükleyin.")
    sys.exit(1)

# --- Python Versiyon Uyarısı ---
def check_python_version():
    major, minor = sys.version_info[:2]
    if major != 3 or minor != 7:
        print("="*60)
        print(" UYARI: Bu script Python 3.7.x ile test edilmiştir! ")
        print(f" Mevcut Python versiyonunuz: {major}.{minor} ")
        print(" Farklı bir versiyon uyumluluk sorunlarına neden olabilir. ")
        print(" Devam etmek için Enter'a basın veya işlemi iptal etmek için Ctrl+C yapın. ")
        print("="*60)
        try:
            input("...") # Kullanıcının görmesi için bekle
        except KeyboardInterrupt:
            print("\nİşlem iptal edildi.")
            sys.exit(1)

check_python_version() # Script başında kontrolü çalıştır
# ------------------------------

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ayarlar ve Sabitler ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # Bir üst dizine çıkıldı
DATASET_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'dataset')
MODELS_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'models')

if not os.path.exists(DATASET_DIR):
    logger.error(f"Veri seti dizini bulunamadı: {DATASET_DIR}")
    sys.exit(1)
if not os.path.exists(MODELS_DIR):
    logger.info(f"Model dizini oluşturuluyor: {MODELS_DIR}")
    os.makedirs(MODELS_DIR)

# --- Veri Seti Sınıfı (modeltrain.py'den alındı) ---
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(self.root_dir):
             raise FileNotFoundError(f"Veri seti dizini bulunamadı: {self.root_dir}")
             
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            raise ValueError(f"Veri seti dizininde sınıf klasörü bulunamadı: {self.root_dir}")
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        logger.info(f"Bulunan sınıflar: {self.classes}")
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            img_count = 0
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
                    img_count += 1
            logger.info(f"  '{cls}' sınıfı için {img_count} görüntü bulundu.")
        if not self.images:
            raise ValueError(f"Veri seti dizininde geçerli görüntü bulunamadı: {self.root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Görüntü açılamadı: {img_path}, Hata: {e}")
            return torch.zeros(3, 160, 160), -1 # FaceNet için boyutlar (160x160) ve geçersiz etiket
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Yardımcı Fonksiyon: Model Kaydetme Yolu (modeltrain.py'den alındı) ---
def _get_next_version_path(base_dir, base_filename, extension):
    base_path = os.path.join(base_dir, f"{base_filename}.{extension}")
    if not os.path.exists(base_path):
        return base_path
    version = 1
    while True:
        versioned_path = os.path.join(base_dir, f"{base_filename}_v{version}.{extension}")
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1

# --- FaceNet Eğitme Fonksiyonu ---
def train_facenet(epochs=50, batch_size=32, learning_rate=0.001):
    """Veri setini kullanarak FaceNet (InceptionResnetV1) modelini sınıflandırma için eğitir."""
    logger.info(f"FaceNet Eğitimi Başlatılıyor: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}")

    try:
        # Veri seti yükleme ve transformasyonlar (FaceNet için 160x160)
        input_size = (160, 160)
        transform = T.Compose([
            T.Resize((180, 180)), # Hafif büyütme
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(), # Yüz verisinde genellikle kullanılmaz
            T.ToTensor(),
            # FaceNet genellikle kendi normalizasyonunu yapar veya [-1, 1] aralığı bekler
            # Önceden eğitilmiş modelin beklentisine göre ayarlamak gerekebilir.
            # Şimdilik standart ImageNet normalizasyonunu kullanalım:
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FaceDataset(DATASET_DIR, transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # Performans için eklemeler
        
        num_classes = len(dataset.classes)
        if num_classes < 2:
             logger.error(f"Eğitim için en az 2 sınıf (kişi) gereklidir. Bulunan: {num_classes}")
             return
             
        # FaceNet modelini yükle ve sınıflandırıcıyı ayarla
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
        logger.info("FaceNet (InceptionResnetV1) modeli oluşturuldu (pretrained='vggface2', classify=True).")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Eğitim için kullanılacak cihaz: {device}")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        logger.info(f"Eğitim döngüsü başlıyor ({epochs} epoch)...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for i, (images, labels) in enumerate(train_loader):
                valid_indices = labels != -1
                if not valid_indices.any(): continue
                images = images[valid_indices].to(device)
                labels = labels[valid_indices].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Epoch sonu istatistikler
            if total_samples > 0:
                epoch_loss = running_loss / total_samples
                epoch_acc = (correct_predictions / total_samples) * 100
                train_losses.append(epoch_loss)
                logger.info(f'Epoch [{epoch+1}/{epochs}] tamamlandı, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            else:
                logger.warning(f"Epoch [{epoch+1}/{epochs}] içinde geçerli örnek bulunamadı.")
        
        logger.info("Eğitim döngüsü tamamlandı.")
        
        # Sürümlü Kaydetme
        base_filename = f"{epochs}_facenet"
        model_path = _get_next_version_path(MODELS_DIR, base_filename, 'pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model kaydedildi: {model_path}")
        
        graph_filename = os.path.splitext(os.path.basename(model_path))[0]
        graph_path = os.path.join(MODELS_DIR, f'{graph_filename}_loss.png')
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title(f'Eğitim Kaybı (FaceNet - {epochs} Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(graph_path)
        plt.close()
        logger.info(f"Eğitim grafiği kaydedildi: {graph_path}")
        
        class_names_filename = f'{graph_filename}_classes.json'
        class_names_path = os.path.join(MODELS_DIR, class_names_filename)
        with open(class_names_path, 'w') as f:
            json.dump(dataset.classes, f)
        logger.info(f"Sınıf isimleri kaydedildi: {class_names_path}")
        
        logger.info("FaceNet modeli başarıyla eğitildi!")
        
    except FileNotFoundError as e:
        logger.error(f"Veri seti hatası: {e}")
    except ValueError as e:
        logger.error(f"Veri hatası: {e}")
    except Exception as e:
        logger.exception("FaceNet model eğitimi sırasında beklenmedik bir hata oluştu:")

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FaceNet (InceptionResnetV1) modeli eğitme scripti.')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim için epoch sayısı.')
    parser.add_argument('--batch_size', type=int, default=32, help='Eğitim için batch boyutu.')
    parser.add_argument('--lr', type=float, default=0.001, help='Eğitim için öğrenme oranı.')
    
    args = parser.parse_args()
    
    train_facenet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
