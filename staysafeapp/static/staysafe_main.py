import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from static.Database_Utils import WorkersDatabase # Opsiyonel

label_to_name = {
    0: "Emre",  # 0 etiketinin ismi
    1: "Kubilay"    # 1 etiketinin ismi
}

db = WorkersDatabase(db_name="Workers.db")

class StaySafe():
    def __init__(self, Model_Name: str, face_model_path: str, db_name, width = 640, height = 640):
        self.Model_Name = Model_Name
        self.face_model_name = face_model_path
        self.width = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.CreateYoloModel()
        self.face_model = self.CreateFaceRecognitionModel()
        self.face_detector = self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.predicted_names = []
        self.database = db_name
        
        # CUDA optimizasyonları
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
    def CreateFaceRecognitionModel(self):
        face_model = torch.load(self.face_model_name, weights_only=False, map_location=self.device)
        face_model.to(self.device)
        face_model.eval()
        return face_model
    
    def CreateYoloModel(self): 
        # Çalışıyor
        """
        Aynı dizinde olan modeli parametre olarak verebiliriz. (Geliştirilecek)
        """
        model = YOLO(self.Model_Name)
        return model
    
    def recognize(self, img, labels=label_to_name):
        self.predicted_names = []  # 🔹 Önceki isimleri temizle
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

        for face in faces:
            
            x, y, w, h = face
            face_roi = img[y:y+h, x:x+w]

            resized = cv2.resize(face_roi, (128, 128))
            normalize = resized / 255.0
            tensor_image = torch.tensor(normalize, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                result = self.face_model(tensor_image)

            _, label = torch.max(result, 1)
            predicted_label = label[0].item()
            predicted_name = labels.get(predicted_label, "Bilinmeyen")
            self.predicted_names.append(predicted_name)  # 🔹 Listeye ekleme

        return img, self.predicted_names  # 🔹 Liste olarak dön
    
    def findWorker(self):
        if not self.predicted_names:
            return ["Boyle bir calisan bulunamadi."]  # 🔹 Boş liste yerine anlamlı dönüş
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
                        self.recognize(frame, label_to_name)
                        worker = self.findWorker()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'{worker}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'Unsafe', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Frame'i JPEG formatına dönüştür
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Django StreamingHttpResponse için yield kullan
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
