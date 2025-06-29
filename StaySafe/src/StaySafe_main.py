import cv2
import imutils
import numpy as np
import os
import torch
import logging
import queue
import threading
import time
from imutils.video import FPS
from ultralytics import YOLO
from Database_Utils import WorkersDatabase
from face_recognizer import FaceRecognitionSystem
from concurrent.futures import ThreadPoolExecutor

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ana dizin
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

db = WorkersDatabase(db_name="Workers.db")

class FrameProcessor:
    def __init__(self, frame_queue, result_queue, model, width=640):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model
        self.width = width
        self.running = False
        self.thread = None
        self.last_result = None
        self.last_result_time = 0
        self.result_cache_time = 0.1  # 100ms cache süresi

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True  # Ana program kapanınca thread de kapanır
        self.thread.start()

    def stop(self):
        self.running = False
        # Thread'i zorla sonlandır
        self.thread = None
        
        # Kuyrukları temizle
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                pass

    def _process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Frame'i yeniden boyutlandır
                frame = imutils.resize(frame, width=self.width)
                
                # Son sonuç çok yeniyse, onu kullan
                current_time = time.time()
                if self.last_result and (current_time - self.last_result_time) < self.result_cache_time:
                    self.result_queue.put((frame, self.last_result))
                    continue
                
                # YOLO modelini çalıştır
                results = self.model(frame, verbose=False)
                self.last_result = results
                self.last_result_time = current_time
                
                self.result_queue.put((frame, results))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Frame işleme hatası: {e}")

class StaySafe():
    def __init__(self, Model_Name: str, db_name, width = 640, height = 480):
        self.Model_Name = Model_Name
        self.width = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.CreateYoloModel()
        self.predicted_names = []
        self.database = db_name
        
        # Yüz tanıma sistemini başlat
        self.face_recognizer = FaceRecognitionSystem()
        
        # CUDA optimizasyonları
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Frame ve sonuç kuyrukları
        self.frame_queue = queue.Queue(maxsize=4)  # Max 4 frame buffer
        self.result_queue = queue.Queue(maxsize=4)
        
        # Frame işleyici
        self.frame_processor = FrameProcessor(
            self.frame_queue,
            self.result_queue,
            self.model,
            self.width
        )
    
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
    
    def process_detection_results(self, frame, results):
        """Tespit sonuçlarını işle ve görüntüye çiz"""
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
                # Yüz bölgesini kes
                face_roi = frame[y1:y2, x1:x2]
                
                # Yüz tanıma işlemini yap
                name, confidence = self.face_recognizer.recognize_faces(face_roi)
                
                # Kırmızı kutu ve bilgileri çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                          (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                self.predicted_names = [name]
        
        return frame
    
    def SafetyDetector(self, recognition=False):
        # Kamera başlatma işlemini face_recognizer üzerinden yap
        if self.face_recognizer.cam is None:
            self.face_recognizer.initialize_camera()
            if self.face_recognizer.cam is None:
                logger.error("Kamera başlatılamadı")
                return
        
        cap = self.face_recognizer.cam
        fps = FPS().start()
        
        # Frame işleyiciyi başlat
        self.frame_processor.start()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                
                # Frame'i kuyruğa ekle
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    # Kuyruk doluysa en eski frame'i at
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except:
                        pass
                
                try:
                    # İşlenmiş frame'i al
                    processed_frame, results = self.result_queue.get(timeout=0.1)
                    
                    # Sonuçları işle ve görüntüye çiz
                    if recognition:
                        processed_frame = self.process_detection_results(processed_frame, results)
                    
                    # FPS'i güncelle ve göster
                    fps.update()
                    fps.stop()
                    cv2.putText(processed_frame, f"FPS: {int(fps.fps())}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Result', processed_frame)
                except queue.Empty:
                    continue
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            # Önce frame işleyiciyi durdur
            self.frame_processor.stop()
            
            # Sonra kamerayı ve pencereleri kapat
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Pencereyi tamamen kapatmak için ekstra waitKey

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
    stay_safe.SafetyDetector(recognition=True)