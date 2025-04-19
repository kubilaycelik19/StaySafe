# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ayarlar ve Sabitler ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'dataset')
MODELS_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'models')

if not os.path.exists(DATASET_DIR):
    logger.error(f"Veri seti dizini bulunamadı: {DATASET_DIR}")
    sys.exit(1)
if not os.path.exists(MODELS_DIR):
    logger.info(f"Model dizini oluşturuluyor: {MODELS_DIR}")
    os.makedirs(MODELS_DIR)

# --- Model ve Veri Seti Sınıfları ---

class ArcFaceModel(nn.Module):
    # ... (İçerik aynı kalacak) ...
    def __init__(self, num_classes, embedding_dim=512):
        super(ArcFaceModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, embedding_dim)
        )
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        features = self.backbone(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        weight = nn.functional.normalize(self.weight, p=2, dim=1)
        cos = nn.functional.linear(features, weight)
        return cos

class FaceDataset(Dataset):
    # ... (İçerik aynı kalacak) ...
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
            return torch.zeros(3, 224, 224), -1 # ArcFace için boyutlar
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Yardımcı Fonksiyon: Model Kaydetme Yolu ---
def _get_next_version_path(base_dir, base_filename, extension):
    # ... (İçerik aynı kalacak) ...
    base_path = os.path.join(base_dir, f"{base_filename}.{extension}")
    if not os.path.exists(base_path):
        return base_path
    version = 1
    while True:
        versioned_path = os.path.join(base_dir, f"{base_filename}_v{version}.{extension}")
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1

# --- ArcFace Eğitme Fonksiyonu ---
def train_arcface(epochs=50, batch_size=32, learning_rate=0.001):
    """Veri setini kullanarak ArcFace modelini eğitir."""
    logger.info(f"Eğitim başlatılıyor: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}")

    try:
        # Veri seti yükleme ve transformasyonlar
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FaceDataset(DATASET_DIR, transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Model oluşturma
        num_classes = len(dataset.classes)
        if num_classes < 2:
             logger.error(f"Eğitim için en az 2 sınıf (kişi) gereklidir. Bulunan: {num_classes}")
             return
             
        model = ArcFaceModel(num_classes=num_classes)
        
        # Eğitim parametreleri
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Eğitim için kullanılacak cihaz: {device}")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Eğitim döngüsü
        train_losses = []
        logger.info(f"Eğitim döngüsü başlıyor ({epochs} epoch)...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # Hatalı görüntüleri atla (-1 etiketli olanlar)
                valid_indices = labels != -1
                if not valid_indices.any(): continue
                images = images[valid_indices].to(device)
                labels = labels[valid_indices].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0) # batch size ile çarp
            
            epoch_loss = running_loss / len(dataset) # Toplam görüntü sayısına böl
            train_losses.append(epoch_loss)
            logger.info(f'Epoch [{epoch+1}/{epochs}] tamamlandı, Loss: {epoch_loss:.4f}')
        
        logger.info("Eğitim döngüsü tamamlandı.")
        
        # Modeli kaydet
        model_path = os.path.join(MODELS_DIR, f'arcface_model_{epochs}e_{learning_rate}lr_{batch_size}bs.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model kaydedildi: {model_path}")
        
        # Eğitim grafiğini kaydet
        graph_path = os.path.join(MODELS_DIR, f'training_loss_{epochs}e_{learning_rate}lr_{batch_size}bs.png')
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Eğitim Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(graph_path)
        plt.close()
        logger.info(f"Eğitim grafiği kaydedildi: {graph_path}")
        
        # Sınıf isimlerini kaydet
        class_names_path = os.path.join(MODELS_DIR, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(dataset.classes, f)
        logger.info(f"Sınıf isimleri kaydedildi: {class_names_path}")
        
        logger.info("Model başarıyla eğitildi!")
        
    except FileNotFoundError as e:
        logger.error(f"Veri seti hatası: {e}")
    except ValueError as e:
        logger.error(f"Veri hatası: {e}")
    except Exception as e:
        logger.exception("Model eğitimi sırasında beklenmedik bir hata oluştu:")

# --- Ana Çalıştırma Bloğu (Doğrudan çalıştırma için) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArcFace modeli eğitme scripti.')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim için epoch sayısı.')
    parser.add_argument('--batch_size', type=int, default=32, help='Eğitim için batch boyutu.')
    parser.add_argument('--lr', type=float, default=0.001, help='Eğitim için öğrenme oranı.')
    
    args = parser.parse_args()
    
    train_arcface(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr) 