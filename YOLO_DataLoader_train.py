import os
import torch
import torchvision.transforms.v2 as T  # Transform v2
from ultralytics import YOLO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatında okur, RGB'ye çevirme islemi.
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # [H, W, C] → [C, H, W]

        # Label'ı oku
        labels = np.loadtxt(label_path).reshape(-1, 5)  # [class_id, x_center, y_center, width, height]
        labels = torch.tensor(labels, dtype=torch.float32)

        # Transform paramametresi
        if self.transform:
            img = self.transform(img)

        return img, labels

##########
# Transform v2 pipeline
#! Makalelere veya chatgpt'den alınan tavsiyelere göre pipelinede değişiklikler yapılmalı.
transform_v2 = T.Compose([
    T.Resize((640, 640)),  # YOLO için standart boyut
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Farklı değerler için toolslar var. Farklı şekillerde denemeler lazım.
    T.ToDtype(torch.float32, scale=True),  # Normalize eder. (Eğitim için hız optimizasyonu)
])
##########

##########
# Dataset oluştur. (Dosya yollarını kendine göre ayarla)
train_dataset = StaySafeDataset(
    img_dir="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/train/images",
    label_dir="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/train/labels",
    transform=transform_v2
)
valid_dataset = StaySafeDataset(
    img_dir="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/valid/images",
    label_dir="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/valid/labels",
    transform=transform_v2
)
test_dataset = StaySafeDataset(
    img_dir="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/test/images",
    label_dir="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/test/labels",
    transform=transform_v2
)
##########
#Dataset objesi oluşturur. Bunlarla eğitim yapılabilir.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) 
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True) 
len(train_loader) # Eğitim için kaç batch olduğunun çıktısını verir
##########

##########
# Eğitim için kullanılacak aygıt. Eğitimden önce kontrol edilmeli, eğer cpu çıktısı alınıyor ise cuda yeniden kurulmalı.
# 1- Conda prompt ekranında: conda activate mevcutenvismi yazılıp kullanılacak environment aktif et.
# 2- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
##########

##########
# Model seçimi ve eğitim kısmı. Aşağıdaki kodlarla temel seviyede eğitim yapıldı.
# Daha fazla ve farklı parametreler kullanılarak (hiperparametre optimizasyonu) eğitim yapılmalı.
model = YOLO('yolov8n.pt') # model tanımlama işlemi. Farklı yolo versiyonları ve versiyonların farklı modelleri (n, m, l, xl) ile eğitim yapılmalı
model.to(device)
if __name__ == "__main__":
    model.train(data="C:/Users/celik/Desktop/ss/helmet_Dataset_Demo/data.yaml", 
    epochs=50, 
    batch=8, 
    device='cuda', 
    name="torch_format_train2"),
    workers=0, # Çoklu işlem hatası için
##########



