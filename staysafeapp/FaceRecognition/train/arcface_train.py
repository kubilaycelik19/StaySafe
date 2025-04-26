# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.transforms import v2 as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import copy
from tqdm import tqdm


# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Metrikler için sklearn (opsiyonel)
try:
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn kütüphanesi yüklü değil. Hassasiyet, Duyarlılık ve F1-Skor metrikleri hesaplanamayacak.")
    SKLEARN_AVAILABLE = False


# --- Ayarlar ve Sabitler ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'dataset')
MODELS_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'models')

# Sabit model ismi (en iyi model için)
BEST_MODEL_FILENAME = 'best_arcface_model.pth'
CLASS_NAMES_FILENAME = 'class_names.json'
TRAINING_PLOT_FILENAME = 'training_history.png'

if not os.path.exists(DATASET_DIR):
    logger.error(f"Veri seti dizini bulunamadı: {DATASET_DIR}")
    sys.exit(1)
if not os.path.exists(MODELS_DIR):
    logger.info(f"Model dizini oluşturuluyor: {MODELS_DIR}")
    os.makedirs(MODELS_DIR)

# --- Model ve Veri Seti Sınıfları ---

class ArcFaceResNetModel(nn.Module):
    """Önceden eğitilmiş ResNet omurgası ve ArcFace benzeri sınıflandırma katmanı."""
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super(ArcFaceResNetModel, self).__init__()
        # Önceden eğitilmiş ResNet18 modelini yükle
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # ResNet'in son sınıflandırma katmanını (fc) çıkar
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Son katmanı etkisiz hale getir

        # Yeni katmanlar: Embedding ve ArcFace benzeri sınıflandırma
        self.embedding_layer = nn.Linear(num_ftrs, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(p=0.5) # Dropout katmanı eklendi

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # ArcFace benzeri ağırlık parametresi
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        features = self.backbone(x) # (batch_size, num_ftrs)
        embedded_features = self.embedding_layer(features) # (batch_size, embedding_dim)
        embedded_features = self.bn(embedded_features) # Batch normalization uygula
        embedded_features = self.dropout(embedded_features) # Dropout uygula (sadece eğitimde aktif olur model.train() ile)

        # Özellikleri ve sınıflandırma ağırlıklarını normalize et
        normalized_features = nn.functional.normalize(embedded_features, p=2, dim=1)
        normalized_weight = nn.functional.normalize(self.weight, p=2, dim=1)

        # Cosine similarity hesapla (logitler)
        cos_theta = nn.functional.linear(normalized_features, normalized_weight)
        return cos_theta # Sınıflandırma için logitler

class FaceDataset(Dataset):
    """Yüz veri setini yükler ve etiketler."""
    def __init__(self, root_dir, class_to_idx=None, transform=None, image_paths=None, labels=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.classes = sorted(list(class_to_idx.keys())) if class_to_idx else None

        # Eğer image_paths ve labels dışarıdan verilmediyse, root_dir'den oku
        if self.image_paths is None or self.labels is None or self.class_to_idx is None:
            if not os.path.exists(self.root_dir):
                raise FileNotFoundError(f"Veri seti dizini bulunamadı: {self.root_dir}")
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            if not self.classes:
                raise ValueError(f"Veri seti dizininde sınıf klasörü bulunamadı: {self.root_dir}")
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.image_paths = []
            self.labels = []
            logger.info(f"Bulunan sınıflar: {self.classes}")
            for cls in self.classes:
                cls_dir = os.path.join(root_dir, cls)
                img_count = 0
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])
                        img_count += 1
                logger.info(f"  '{cls}' sınıfı için {img_count} görüntü bulundu.")
            if not self.image_paths:
                raise ValueError(f"Veri seti dizininde geçerli görüntü bulunamadı: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Görüntü açılamadı: {img_path}, Hata: {e}")
            # Hatalı görüntü yerine boş tensor ve geçersiz etiket döndür
            return torch.zeros(3, 224, 224), -1
        label = self.labels[idx]
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 logger.error(f"Transform uygulanamadı: {img_path}, Hata: {e}")
                 return torch.zeros(3, 224, 224), -1
        return image, label

# --- Veri Dönüşümleri ---
IMG_SIZE = 224
# Eğitim için veri artırma ile
data_transforms = {
    'train': T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)), # Önce biraz büyüt
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Renk artırma
        T.RandomHorizontalFlip(),
        T.RandomRotation(15), # İsteğe bağlı ek rotasyon
        # T.ToTensor(), # Eski yöntem 
        T.ToImage(), # PIL veya NumPy'dan Tensor'e çevirir
        T.ToDtype(torch.float32, scale=True), # float32'ye çevirir ve [0, 1] aralığına ölçekler
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Doğrulama için sadece temel dönüşümler
    'val': T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        # T.ToTensor(), # Eski yöntem
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# --- Yardımcı Fonksiyon: Eğitim Geçmişi Grafiği ---
def plot_training_history(history, save_path):
    """Eğitim ve doğrulama loss/accuracy grafiklerini çizer ve kaydeder."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Eğitim Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Doğrulama Loss')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Eğitim Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Doğrulama Accuracy')
    plt.title('Eğitim ve Doğrulama Başarısı')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Eğitim grafiği kaydedildi: {save_path}")


# --- ArcFace Eğitme Fonksiyonu (İyileştirilmiş) ---
def train_arcface(epochs=50, batch_size=32, learning_rate=0.001, val_split=0.2,
                  scheduler_step=10, scheduler_gamma=0.1, weight_decay=1e-4, # weight_decay eklendi
                  early_stopping_patience=5, early_stopping_delta=0.001): # early stopping parametreleri eklendi
    """Veri setini kullanarak iyileştirilmiş ArcFace modelini eğitir."""
    logger.info(
        f"İyileştirilmiş Eğitim Başlatılıyor: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}, "
        f"Val Split={val_split}, Weight Decay={weight_decay}, EarlyStop Patience={early_stopping_patience}"
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Eğitim için kullanılacak cihaz: {device}")

        # 1. Veri Setini Yükle ve Böl
        full_dataset = FaceDataset(DATASET_DIR) # Dönüşümleri daha sonra uygulayacağız
        num_classes = len(full_dataset.classes)
        if num_classes < 2:
            logger.error(f"Eğitim için en az 2 sınıf (kişi) gereklidir. Bulunan: {num_classes}")
            return

        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        if train_size == 0 or val_size == 0:
            logger.error(f"Veri seti boyutu ({dataset_size}) eğitim/doğrulama bölmesi için çok küçük.")
            # Sadece eğitim yapmayı dene? Veya hata ver. Şimdilik hata verelim.
            return

        train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size])

        # Subset'leri oluştururken ilgili transformları ata
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # ÖNEMLİ: Subset'lerin içindeki __getitem__ transformu dinamik olarak atamak gerekiyor.
        # Ya da FaceDataset'i Subset'leri destekleyecek şekilde modifiye etmeli.
        # Şimdilik daha basit yol: Subset yerine yeni FaceDataset nesneleri oluşturmak.
        train_dataset = FaceDataset(
            root_dir=DATASET_DIR,
            class_to_idx=full_dataset.class_to_idx,
            transform=data_transforms['train'],
            image_paths=[full_dataset.image_paths[i] for i in train_indices],
            labels=[full_dataset.labels[i] for i in train_indices]
        )
        val_dataset = FaceDataset(
            root_dir=DATASET_DIR,
            class_to_idx=full_dataset.class_to_idx,
            transform=data_transforms['val'],
            image_paths=[full_dataset.image_paths[i] for i in val_indices],
            labels=[full_dataset.labels[i] for i in val_indices]
        )


        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4), # num_workers 4'e çıkarıldı
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) # num_workers 4'e çıkarıldı
        }
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
        logger.info(f"Veri seti bölündü: Eğitim={dataset_sizes['train']}, Doğrulama={dataset_sizes['val']}")

        # 2. Modeli Oluştur
        model = ArcFaceResNetModel(num_classes=num_classes, pretrained=True).to(device)

        # 3. Optimizasyon Ayarları
        criterion = nn.CrossEntropyLoss()
        # Tüm modeli eğitmek (fine-tuning) genellikle daha iyi sonuç verir.
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # weight_decay eklendi
        # Öğrenme oranı zamanlayıcısı
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        # 4. Eğitim Döngüsü
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_loss = float('inf') # En iyi doğrulama kaybını takip et
        best_epoch = -1 # En iyi epoch numarasını sakla
        best_val_acc_at_best_loss = 0.0 # En iyi kayıp anındaki doğruluğu sakla
        epochs_no_improve = 0 # İyileşme olmayan epoch sayacı
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        logger.info(f"Eğitim döngüsü başlıyor ({epochs} epoch)...")
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}/{epochs}')
            logger.info('-' * 10)

            # Her epoch için eğitim ve doğrulama fazı
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Modeli eğitim moduna al
                else:
                    model.eval()   # Modeli değerlendirme moduna al

                running_loss = 0.0
                running_corrects = 0

                # Veri üzerinde iterate et (tqdm ile)
                pbar = tqdm(dataloaders[phase], desc=f'Epoch {epoch+1}/{epochs} {phase.capitalize()}', leave=False) # leave=False iç içe barlar için
                batch_losses = [] # Anlık loss için
                for inputs, labels in pbar:
                     # Hatalı yüklenenleri atla
                    valid_indices = labels != -1
                    if not valid_indices.any(): continue
                    inputs = inputs[valid_indices].to(device)
                    labels = labels[valid_indices].to(device)

                    # Parametre gradyanlarını sıfırla
                    optimizer.zero_grad()

                    # İleri geçiş
                    # Sadece eğitim fazında gradyanları izle
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Sadece eğitim fazında geri yayılım ve optimize et
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # İstatistikler
                    batch_loss = loss.item()
                    batch_losses.append(batch_loss)
                    running_loss += batch_loss * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # tqdm ilerleme çubuğuna anlık loss'u ekle
                    pbar.set_postfix(loss=f'{batch_loss:.4f}')

                # Epoch sonunda öğrenme oranını ayarla (eğitim fazıysa)
                if phase == 'train':
                    exp_lr_scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"  Öğrenme Oranı: {current_lr:.7f}")


                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc.item()) # Tensor'den float'a çevir

                logger.info(f'  {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Modeli kopyala (en iyi ise) - sadece doğrulama fazında, kayba göre
                if phase == 'val':
                    if epoch_loss < best_val_loss - early_stopping_delta:
                        best_val_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        best_epoch = epoch + 1 # Epoch 1-tabanlı
                        best_val_acc_at_best_loss = epoch_acc.item() # Doğruluğu sakla
                        logger.info(f"  ** Yeni en iyi doğrulama kaybı: {best_val_loss:.4f} (Acc: {epoch_acc:.4f}) Epoch: {best_epoch} **")
                    else:
                        epochs_no_improve += 1
                        logger.info(f"  Doğrulama kaybı iyileşmedi ({epochs_no_improve}/{early_stopping_patience})")

            # Early stopping kontrolü (her epoch sonunda)
            if epochs_no_improve >= early_stopping_patience:
                logger.warning(f"Early stopping tetiklendi! {early_stopping_patience} epoch boyunca doğrulama kaybı iyileşmedi.")
                print('\a') # Bitiş sinyali (Early Stopping)
                break # Eğitim döngüsünü sonlandır

        logger.info("Eğitim döngüsü tamamlandı.")
        logger.info(f'En iyi doğrulama kaybı: {best_val_loss:4f} (Epoch: {best_epoch}, Acc: {best_val_acc_at_best_loss:.4f})')

        # 5. En İyi Modeli Yükle (Devam etmeden önce)
        if best_epoch == -1:
            logger.warning("Hiçbir epoch'ta iyileşme kaydedilmediği için değerlendirme ve kaydetme adımları atlanıyor.")
        else:
            model.load_state_dict(best_model_wts) # En iyi ağırlıkları yükle

            # ---------------------------------------------------
            # 5.1 En İyi Model ile Son Değerlendirme (Doğrulama Seti) - KAYDETMEDEN ÖNCE
            # ---------------------------------------------------
            logger.info("En iyi model ile doğrulama seti üzerinde son değerlendirme yapılıyor...")
            model.eval() # Değerlendirme moduna al
            all_labels = []
            all_preds = []
            final_val_loss = 0.0
            # final_running_corrects = 0 # Artık accuracy_score kullanıldığı için gereksiz

            with torch.no_grad(): # Gradyan hesaplamayı kapat
                pbar_eval = tqdm(dataloaders['val'], desc='Son Değerlendirme')
                for inputs, labels in pbar_eval:
                    valid_indices = labels != -1
                    if not valid_indices.any(): continue
                    inputs = inputs[valid_indices].to(device)
                    labels = labels[valid_indices].to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    final_val_loss += loss.item() * inputs.size(0)
                    # final_running_corrects += torch.sum(preds == labels.data)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            final_val_loss /= dataset_sizes['val']

            logger.info("--- Değerlendirme Sonuçları (En İyi Model) ---")
            logger.info(f"Doğrulama Kaybı (Loss): {final_val_loss:.4f}")

            # Metrikleri hesapla ve kaydetme koşulunu kontrol et
            should_save_model = True
            metrics_report_str = "Metrikler hesaplanamadı."
            final_accuracy = 0.0

            if SKLEARN_AVAILABLE and len(all_labels) > 0:
                try:
                    final_accuracy = accuracy_score(all_labels, all_preds)
                    logger.info(f"Doğrulama Başarısı (Accuracy): {final_accuracy:.4f}")

                    # Sınıf isimlerini al (target_names için)
                    class_names = full_dataset.classes
                    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
                    metrics_report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4) # String formatı loglama için
                    logger.info("\nClassification Report:\n" + metrics_report_str)

                    # Metrikleri kontrol et (> 0.995)
                    metrics_to_check = {
                        'accuracy': final_accuracy,
                        'macro_precision': report_dict['macro avg']['precision'],
                        'macro_recall': report_dict['macro avg']['recall'],
                        'macro_f1': report_dict['macro avg']['f1-score'],
                        'weighted_precision': report_dict['weighted avg']['precision'],
                        'weighted_recall': report_dict['weighted avg']['recall'],
                        'weighted_f1': report_dict['weighted avg']['f1-score']
                    }

                    overfitting_metrics = []
                    for metric_name, metric_value in metrics_to_check.items():
                        if metric_value > 0.995:
                            overfitting_metrics.append(f"{metric_name}: {metric_value:.4f}")

                    if overfitting_metrics:
                        should_save_model = False
                        logger.warning("\nMODEL KAYDEDİLMEDİ! Aşağıdaki metrik(ler) %99.5 sınırını aştı (overfitting riski):")
                        for item in overfitting_metrics:
                            logger.warning(f"  - {item}")
                        logger.warning("-----------------------------------------------------------------------")

                except Exception as report_err:
                    logger.error(f"Classification report oluşturulurken veya kontrol edilirken hata: {report_err}")
                    should_save_model = False # Hata durumunda kaydetme
                    # Accuracy hala hesaplanmış olabilir, onu logla
                    try:
                        final_accuracy = accuracy_score(all_labels, all_preds)
                        logger.info(f"Doğrulama Başarısı (Accuracy): {final_accuracy:.4f}")
                    except: pass
            elif len(all_labels) == 0:
                 logger.warning("Değerlendirme için geçerli etiket bulunamadı. Model kaydedilmeyecek.")
                 should_save_model = False
            else: # sklearn yoksa
                 logger.info("Detaylı metrik kontrolü için scikit-learn gerekli. Model kaydedilecek (sadece kayıp bazında). ")
                 # should_save_model True kalır (ya da burada False yapıp sadece Loss'a göre kaydetmeyi engellersiniz)

            logger.info("----------------------------------------------")

            # 5.2 Modeli Kaydet (Eğer koşullar sağlandıysa)
            if should_save_model:
                # Dinamik dosya adını oluştur
                model_name = "ArcFaceResNet"
                accuracy_str = f"acc{final_accuracy*100:.1f}"
                dynamic_model_filename = f"{model_name}_epoch{best_epoch}_bs{batch_size}_{accuracy_str}.pth"
                dynamic_model_path = os.path.join(MODELS_DIR, dynamic_model_filename)

                # Sabit dosya adını belirle
                fixed_model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILENAME) # BEST_MODEL_FILENAME = 'best_arcface_model.pth'

                # Modeli hem dinamik hem de sabit isimle kaydet
                torch.save(model.state_dict(), dynamic_model_path)
                torch.save(model.state_dict(), fixed_model_path) # Sabit dosyanın üzerine yaz

                logger.info(f"Model başarıyla kaydedildi:")
                logger.info(f"  Dinamik İsim: {dynamic_model_path}")
                logger.info(f"  Sabit İsim (Uygulama için): {fixed_model_path}")
                print(f"Kaydedilen Model Dosyası (Dinamik): {dynamic_model_path}")
                print(f"Kaydedilen Model Dosyası (Sabit): {fixed_model_path}")
            else:
                 logger.info("Model yukarıda belirtilen nedenlerden dolayı kaydedilmedi.")

        # 6. Eğitim Grafiğini Kaydet
        plot_path = os.path.join(MODELS_DIR, TRAINING_PLOT_FILENAME)
        plot_training_history(history, plot_path)

        # 7. Sınıf İsimlerini Kaydet (Her durumda kaydedilebilir)
        class_names_path = os.path.join(MODELS_DIR, CLASS_NAMES_FILENAME)
        with open(class_names_path, 'w') as f:
            json.dump(full_dataset.classes, f) # Tam veri setinden al
        logger.info(f"Sınıf isimleri kaydedildi: {class_names_path}")

        # logger.info("Model başarıyla eğitildi ve kaydedildi!") # Mesaj kaydetme durumuna göre değişti
        print('\a') # Bitiş sinyali (Normal Bitiş veya Early Stopping sonrası)

    except FileNotFoundError as e:
        logger.error(f"Veri seti hatası: {e}")
    except ValueError as e:
        logger.error(f"Veri hatası: {e}")
    except Exception as e:
        logger.exception("Model eğitimi sırasında beklenmedik bir hata oluştu:")


# --- Ana Çalıştırma Bloğu (Doğrudan çalıştırma için) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='İyileştirilmiş ArcFace modeli eğitme scripti.')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim için epoch sayısı.')
    parser.add_argument('--batch_size', type=int, default=16, help='Eğitim için batch boyutu (GPU belleğine göre ayarlayın).') # Daha küçük başla
    parser.add_argument('--lr', type=float, default=0.001, help='Eğitim için başlangıç öğrenme oranı.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Doğrulama seti için ayrılacak veri oranı (0.0 ile 1.0 arası).')
    parser.add_argument('--scheduler_step', type=int, default=7, help='Öğrenme oranı düşürme adımı (epoch).')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Öğrenme oranı düşürme çarpanı.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Optimizer için weight decay (L2 regularization) değeri.') # weight_decay argümanı eklendi
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping için beklenecek iyileşmesiz epoch sayısı.') # early stopping argümanları eklendi
    parser.add_argument('--early_stopping_delta', type=float, default=0.001, help='Early stopping için minimum iyileşme miktarı (kayıp).')

    args = parser.parse_args()

    # Argümanları kontrol et
    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("val_split 0.0 ile 1.0 arasında olmalıdır.")

    train_arcface(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        weight_decay=args.weight_decay, # Yeni argümanlar eklendi
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta
    ) 