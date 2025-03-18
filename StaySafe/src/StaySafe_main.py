import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from imutils.video import FPS
from ultralytics import YOLO
from Database_Utils import WorkersDatabase
from face_recognizer import FaceRecognitionSystem
from settings.settings import FACE_DETECTION, CAMERA

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_names(names_path):
    """JSON dosyasından isimleri yükler"""
    try:
        with open(names_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"İsimler yüklenirken hata oluştu: {e}")
        return {}

# Ana dizin
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Yüz tanıma modeli yolu
TRAINER_PATH = os.path.join(CURRENT_DIR, "trainer.yml")
NAMES_PATH = os.path.join(CURRENT_DIR, "names.json")

# Yolları kontrol et
logger.info(f"Current directory: {CURRENT_DIR}")
for path in [TRAINER_PATH, NAMES_PATH]:
    if not os.path.exists(path):
        logger.warning(f"UYARI: Dosya bulunamadı: {path}")
    else:
        logger.info(f"OK: Dosya bulundu: {path}")

# Sınıf isimlerini yükle
try:
    names = load_names(NAMES_PATH)
    logger.info(f"İsimler yüklendi: {names}")
except Exception as e:
    logger.error(f"İsimler yüklenirken hata oluştu: {e}")
    names = {}

db = WorkersDatabase(db_name="Workers.db")

class StaySafe():
    def __init__(self, Model_Name: str, db_name, width = 640, height = 480):
        self.Model_Name = Model_Name
        self.width = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.CreateYoloModel()
        self.predicted_names = []
        self.database = db_name
        
        # Yüz tanıma sistemi başlatma
        self.face_recognition = FaceRecognitionSystem()
        
        # CUDA optimizasyonları
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def CreateYoloModel(self): 
        model = YOLO(self.Model_Name)
        return model
    
    def findWorker(self):
        if not self.predicted_names:
            return ["Boyle bir calisan bulunamadi."]
        workers = []
        for name in self.predicted_names:
            try:
                worker = db.find_employee(name=name)
            except:
                worker = "Boyle bir calisan bulunamadi."
            workers.append(worker)
        return workers
    
    def SafetyDetector(self, Source, recognition=False):
        cap = cv2.VideoCapture(Source)
        
        # Kamera ayarlarını yapılandır
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        
        fps = FPS().start()  # FPS sayacı başlat
        
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
                    # Güvenli durum - yeşil kutu
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Safe', (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    if recognition:
                        # Yüz bölgesini kes
                        face_roi = frame[y1:y2, x1:x2]
                        
                        # Gri tonlamaya çevir
                        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        
                        # Yüz tespiti yap
                        faces = self.face_recognition.face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=FACE_DETECTION['scale_factor'],
                            minNeighbors=FACE_DETECTION['min_neighbors'],
                            minSize=FACE_DETECTION['min_size']
                        )
                        
                        for (fx, fy, fw, fh) in faces:
                            # Yüz tanıma yap
                            face_id, confidence = self.face_recognition.recognizer.predict(gray[fy:fy+fh, fx:fx+fw])
                            name = self.face_recognition.names.get(str(face_id), "Unknown")
                            confidence_text = f"{confidence:.1f}%"
                            
                            # Kırmızı kutu ve bilgileri çiz
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, name, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.putText(frame, confidence_text, (x1, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            self.predicted_names = [name]
                    else:
                        # Güvensiz durum - kırmızı kutu
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'Unsafe', (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # FPS'i güncelle ve göster
            fps.update()
            fps.stop()
            cv2.putText(frame, f"FPS: {int(fps.fps())}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Result', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Model yolları
    yolo_model_path = os.path.join(CURRENT_DIR, "Yolo11n_50_epoch.pt")
    
    logger.info(f"YOLO model yolu: {yolo_model_path}")
    if not os.path.exists(yolo_model_path):
        logger.error(f"UYARI: YOLO modeli bulunamadı: {yolo_model_path}")
        logger.error("Lütfen modeli doğru konuma yerleştirin.")
        exit(1)
    
    # StaySafe nesnesini oluştur
    stay_safe = StaySafe(
        Model_Name=yolo_model_path,
        db_name="Workers.db"
    )
    
    # Güvenlik kontrolünü başlat
    stay_safe.SafetyDetector(Source=0, recognition=True)