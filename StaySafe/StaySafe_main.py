import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from imutils.video import FPS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from myUtils.Database_Utils import WorkersDatabase
from torchvision import models, transforms


label_to_name = {
    0: "Emre",  # 0 etiketinin ismi
    1: "Kubilay"    # 1 etiketinin ismi
}

db = WorkersDatabase(db_name="Workers.db")

class StaySafe():
    def __init__(self, Model_Name: str, face_model_path: str, db_name, width = 1280, height = 1280):
        self.Model_Name = Model_Name
        self.face_model_name = face_model_path
        self.width = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.CreateYoloModel()  # PPE tespiti için YOLO
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Yüz tespiti için Haar Cascade
        self.face_model = self.CreateFaceRecognitionModel()  # Yüz tanıma için ResNet18
        self.predicted_names = []
        self.database = db_name
        
        # CUDA optimizasyonları
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
    def CreateFaceRecognitionModel(self):
        """ResNet18 yüz tanıma modelini yükle"""
        try:
            print(f"\nModel yükleme işlemi başlatılıyor: {self.face_model_name}")
            
            # Model dosyasının varlığını kontrol et
            if not os.path.exists(self.face_model_name):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {self.face_model_name}")
            
            # Checkpoint'i yükle
            checkpoint = torch.load(self.face_model_name, map_location=self.device)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint formatı geçersiz. Dictionary bekleniyor.")
            
            # Gerekli anahtarların varlığını kontrol et
            required_keys = ['model_state_dict', 'class_names', 'val_acc']
            if not all(key in checkpoint for key in required_keys):
                raise KeyError(f"Eksik anahtarlar: {[key for key in required_keys if key not in checkpoint]}")
            
            # ResNet18 modelini oluştur
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features  # ResNet18'de 512
            model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
            
            # Model ağırlıklarını yükle
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Sınıf isimlerini kaydet
            self.face_class_names = checkpoint['class_names']
            
            print("\nModel başarıyla yüklendi:")
            print(f"- Model: ResNet18")
            print(f"- Giriş boyutu: 224x224")
            print(f"- Sınıf sayısı: {len(self.face_class_names)}")
            print(f"- Sınıflar: {self.face_class_names}")
            print(f"- Doğrulama başarımı: {checkpoint['val_acc']:.2f}%")
            print(f"- Model mimarisi:")
            print("  * FC katmanları: 512->256->num_classes")
            print("  * Dropout: 0.2")
            
            return model
            
        except Exception as e:
            print(f"\nHATA: Model yüklenirken bir sorun oluştu:")
            print(f"Hata türü: {type(e).__name__}")
            print(f"Hata mesajı: {str(e)}")
            raise
    
    def CreateYoloModel(self): 
        """YOLO modelini yükle"""
        model = YOLO(self.Model_Name)
        return model
    
    def recognize(self, img):
        """Yüz tanıma işlemi"""
        self.predicted_names = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Haar Cascade ile yüz tespiti
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            maxSize=(300, 300)
        )

        if not hasattr(self, 'last_predictions'):
            self.last_predictions = []
        
        for (x, y, w, h) in faces:
            # Yüz bölgesini kes ve işle
            margin = int(w * 0.1)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            
            face_roi = img[y1:y2, x1:x2]
            
            try:
                # Görüntü işleme
                resized = cv2.resize(face_roi, (224, 224))
                lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl,a,b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(enhanced)
                
                tensor_image = normalize.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.face_model(tensor_image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, predicted_label = torch.max(probabilities, 1)
                    
                    if max_prob.item() > 0.5:  # Güven eşiği
                        predicted_label = predicted_label.item()
                        predicted_name = self.face_class_names[predicted_label]
                        
                        # Son tahminleri güncelle
                        self.last_predictions.append(predicted_name)
                        if len(self.last_predictions) > 5:
                            self.last_predictions.pop(0)
                        
                        # Çoğunluk oylaması
                        if len(self.last_predictions) >= 3:
                            from collections import Counter
                            most_common = Counter(self.last_predictions).most_common(1)
                            if most_common[0][1] >= 3:
                                self.predicted_names = [most_common[0][0]]
                                
                                # Yüz bölgesini çiz ve ismi göster
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, predicted_name, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            except Exception as e:
                print(f"Görüntü işleme hatası: {str(e)}")
                continue

        return img, self.predicted_names
    
    def findWorker(self):
        """Veritabanından çalışan bilgilerini al"""
        workers = []
        for name in self.predicted_names:
            worker_info = db.find_employee(name=name)
            if worker_info:
                worker_str = f"{worker_info['name']} {worker_info['surname']} ({worker_info['age']} yas)"
                workers.append(worker_str)
        return workers

    def SafetyDetector(self, Source, recognition=False):
        """Güvenlik kontrolü ve yüz tanıma"""
        cap = cv2.VideoCapture(Source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=self.width)
            
            # Model ile tahmin yap
            results = self.model(frame, verbose=False)
            
            # Sınıf isimlerini al
            class_names = self.model.names
            boxes = results[0].boxes
            
            # Tüm person'ları bul
            persons = [box for box in boxes if class_names[int(box.cls)] == 'person']
            
            # Her bir person için helmet ve vest kontrolü yap
            for person in persons:
                x1, y1, x2, y2 = map(int, person.xyxy[0])
                has_helmet = False
                has_vest = False
                
                for other_box in boxes:
                    other_class_id = int(other_box.cls)
                    other_class_name = class_names[other_class_id]
                    other_x1, other_y1, other_x2, other_y2 = map(int, other_box.xyxy[0])
                    
                    if (other_class_name == 'helmet' or other_class_name == 'vest') and \
                    (other_x1 > x1 and other_x2 < x2 and other_y1 > y1 and other_y2 < y2):
                        if other_class_name == 'helmet':
                            has_helmet = True
                        elif other_class_name == 'vest':
                            has_vest = True
                
                if has_helmet or has_vest:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Safe', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    if recognition:
                        # Yüz tanıma yap
                        self.recognize(frame)
                        workers = self.findWorker()
                        message = workers[0] if workers else "Bilinmiyor"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, message, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'Unsafe', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            cv2.imshow('Result', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Model yolları
    yolo_model_path = "C:/Users/celik/Desktop/ss2/Models/Yolo11n_50_epoch.pt"
    face_model_path = "C:/Users/celik/Desktop/StaySafe/best_face_model.pth"
    
    # StaySafe nesnesini oluştur
    stay_safe = StaySafe(
        Model_Name=yolo_model_path,
        face_model_path=face_model_path,
        db_name="Workers.db"
    )
    
    # Güvenlik kontrolünü başlat
    stay_safe.SafetyDetector(Source=0, recognition=True)