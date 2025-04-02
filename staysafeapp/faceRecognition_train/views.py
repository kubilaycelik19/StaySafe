import cv2
import mediapipe as mp
import os
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import time  # Zaman kontrolü için eklendi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T # v2 eklendi
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe Face Mesh ayarları
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kamera ayarları
CAMERA = {
    'index': 0,
    'width': 640,
    'height': 480
}

# Veri seti ayarları
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'dataset')
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# ArcFace Model Tanımı
class ArcFaceModel(nn.Module):
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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DatasetCreator:
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.current_person = None
        self.frame_count = 0
        self.max_frames = 200  # Her kişi için kaydedilecek maksimum frame sayısı
        self.save_interval_seconds = 0.5  # Kayıt aralığı (saniye)
        self.last_save_time = 0  # Son kayıt zamanı
        self.person_dir = None
        self.face_detected = False

    def initialize_camera(self):
        """Kamerayı başlatır."""
        if self.camera is not None and self.camera.isOpened():
            return True

        try:
            self.camera = cv2.VideoCapture(CAMERA['index'])
            if not self.camera.isOpened():
                logger.error(f"Kamera açılamadı: {CAMERA['index']}")
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            logger.info("Kamera başarıyla başlatıldı.")
            return True
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            return False

    def release_camera(self):
        """Kamerayı serbest bırakır."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.info("Kamera serbest bırakıldı.")

    def create_person_directory(self, person_name):
        """Kişi için veri seti dizini oluşturur."""
        self.current_person = person_name
        self.person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.exists(self.person_dir):
            os.makedirs(self.person_dir)
            logger.info(f"Kişi dizini oluşturuldu: {self.person_dir}")
        self.frame_count = 0
        self.face_detected = False

    def process_frame(self, frame):
        """Frame'i işler ve yüz tespiti yapar."""
        if frame is None:
            return None

        # BGR'den RGB'ye dönüştür
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Yüz tespiti yap
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # İlk yüzü al
            face_landmarks = results.multi_face_landmarks[0]
            
            # Yüz sınırlayıcı kutu hesapla
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Yüz bölgesini genişlet
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Yüz bölgesini çiz
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Eğer kişi dizini varsa ve kayıt aktifse ve zaman aralığı uygunsa
            current_time = time.time()
            if self.person_dir and self.is_active and self.frame_count < self.max_frames and \
               (current_time - self.last_save_time) >= self.save_interval_seconds:
                face_img = frame[y_min:y_max, x_min:x_max]
                if face_img.size > 0:
                    img_path = os.path.join(self.person_dir, f'face_{self.frame_count}.jpg')
                    cv2.imwrite(img_path, face_img)
                    logger.info(f"Yüz kaydedildi: {img_path} ({self.frame_count + 1}/{self.max_frames})")
                    self.face_detected = True
                    self.frame_count += 1
                    self.last_save_time = current_time  # Son kayıt zamanını güncelle
                
            # Frame sayısını göster (kayıt yapılmasa bile)
            if self.person_dir and self.is_active:
                 cv2.putText(frame, f"Kaydedilen: {self.frame_count}/{self.max_frames}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self.face_detected = False
        
        return frame

    def get_video_stream(self):
        """Video akışını üreten generator fonksiyonu."""
        if not self.is_active or self.camera is None or not self.camera.isOpened():
            error_frame = np.zeros((CAMERA['height'], CAMERA['width'], 3), dtype=np.uint8)
            cv2.putText(error_frame, "Kamera Kapalı", (int(CAMERA['width']/2)-100, int(CAMERA['height']/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        while self.is_active:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Frame okunamadı")
                break

            # Frame'i aynala
            frame = cv2.flip(frame, 1)
            
            # Frame'i işle
            processed_frame = self.process_frame(frame)
            
            if processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Global DatasetCreator nesnesi
dataset_creator = DatasetCreator()

def index(request):
    """Ana sayfa view'ı."""
    return render(request, 'faceRecognition_train/index.html')

def train(request):
    """Model eğitim sayfası."""
    context = {
        'dataset_dir': DATASET_DIR
    }
    return render(request, 'faceRecognition_train/train.html', context)

@csrf_exempt
def start_camera(request):
    """Kamerayı başlatır."""
    if request.method == 'POST':
        try:
            if dataset_creator.initialize_camera():
                dataset_creator.is_active = True
                return JsonResponse({'status': 'success', 'message': 'Kamera başlatıldı'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Kamera başlatılamadı'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Geçersiz istek metodu'})

@csrf_exempt
def stop_camera(request):
    """Kamerayı durdurur."""
    if request.method == 'POST':
        try:
            dataset_creator.is_active = False
            dataset_creator.release_camera()
            return JsonResponse({'status': 'success', 'message': 'Kamera durduruldu'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Geçersiz istek metodu'})

@csrf_exempt
def start_recording(request):
    """Yüz kaydını başlatır."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            person_name = data.get('person_name')
            if not person_name:
                return JsonResponse({'status': 'error', 'message': 'Kişi adı gerekli'})
            
            dataset_creator.create_person_directory(person_name)
            return JsonResponse({'status': 'success', 'message': f'{person_name} için kayıt başlatıldı'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Geçersiz istek metodu'})

def video_feed(request):
    """Video akışını sağlar."""
    return StreamingHttpResponse(dataset_creator.get_video_stream(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def train_model(request):
    """ArcFace modelini eğitir."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            num_epochs = data.get('num_epochs', 50)
            batch_size = data.get('batch_size', 32)
            learning_rate = data.get('learning_rate', 0.001)
            
            logger.info(f"Eğitim başlatılıyor: Epochs={num_epochs}, Batch Size={batch_size}, LR={learning_rate}")
            
            # Veri seti yükleme ve transformasyonlar (v2 ile güncellendi)
            transform = T.Compose([
                T.Resize((256, 256)), # Önce biraz büyütelim
                T.RandomResizedCrop(224, scale=(0.8, 1.0)), # Rastgele zoom ve kırpma
                T.RandomHorizontalFlip(), # Rastgele yatay aynalama
                T.RandomVerticalFlip(),   # Rastgele dikey aynalama
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = FaceDataset(DATASET_DIR, transform=transform)
            if len(dataset) == 0:
                 return JsonResponse({'status': 'error', 'message': 'Veri seti boş veya bulunamadı.'})
                 
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Model oluşturma
            num_classes = len(dataset.classes)
            if num_classes == 0:
                return JsonResponse({'status': 'error', 'message': 'Veri setinde sınıf bulunamadı.'})
                
            model = ArcFaceModel(num_classes=num_classes)
            
            # Eğitim parametreleri
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Eğitim için kullanılacak cihaz: {device}")
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Eğitim döngüsü
            # num_epochs = 50 # Yukarıdan alınıyor
            train_losses = []
            last_epoch_loss = 0.0 # Son loss değerini saklamak için
            
            logger.info(f"Eğitim döngüsü başlıyor ({num_epochs} epoch)...")
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # İlerleme loglaması (isteğe bağlı, çok sık olabilir)
                    # if (i+1) % 10 == 0:
                    #     logger.debug(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                
                epoch_loss = running_loss / len(train_loader)
                train_losses.append(epoch_loss)
                last_epoch_loss = epoch_loss # Son epoch loss değerini güncelle
                
                logger.info(f'Epoch [{epoch+1}/{num_epochs}] tamamlandı, Loss: {epoch_loss:.4f}')
            
            logger.info("Eğitim döngüsü tamamlandı.")
            
            # Modeli kaydet
            model_save_dir = os.path.join(DATASET_DIR, '..' , 'models') # Modeli dataset'ten ayrı bir yere kaydedelim
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            model_path = os.path.join(model_save_dir, 'arcface_model.pth') 
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model kaydedildi: {model_path}")
            
            # Eğitim grafiğini kaydet
            graph_path = os.path.join(model_save_dir, 'training_loss.png')
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses)
            plt.title('Eğitim Kaybı')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(graph_path)
            plt.close()
            logger.info(f"Eğitim grafiği kaydedildi: {graph_path}")
            
            # Sınıf isimlerini kaydet (tanıma için gerekli olabilir)
            class_names_path = os.path.join(model_save_dir, 'class_names.json')
            with open(class_names_path, 'w') as f:
                json.dump(dataset.classes, f)
            logger.info(f"Sınıf isimleri kaydedildi: {class_names_path}")
            
            return JsonResponse({
                'status': 'success',
                'message': 'Model başarıyla eğitildi',
                'model_path': model_path,
                'num_classes': num_classes,
                'last_loss': f'{last_epoch_loss:.4f}', # Son loss değerini de gönderelim
                'graph_path': graph_path # Grafik yolunu da gönderelim
            })
            
        except Exception as e:
            logger.exception("Model eğitimi sırasında hata oluştu:") # Hatanın detayını logla
            return JsonResponse({'status': 'error', 'message': f'Model eğitimi sırasında hata: {str(e)}'})
    return JsonResponse({'status': 'error', 'message': 'Geçersiz istek metodu'})
