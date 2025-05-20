import cv2
import imutils
import numpy as np
import os
import torch
import logging
import queue
import threading
import time
import json
import warnings
from ultralytics import YOLO
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from reports.models import EmployeeReport
from django.utils import timezone
from datetime import timedelta
from django.db.models.functions import TruncDate, Concat
from django.db.models import Count, F, Value
from django.contrib.auth.decorators import login_required

# ArcFace için gerekli kütüphaneler
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2 as T
from PIL import Image

# Loglama ayarlarını yapıyorum
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Yüz tanıma yöntemini seçiyorum - ArcFace kullanacağım
FACE_RECOGNITION_METHOD = 'arcface'
logger.info(f"Kullanılacak yüz tanıma yöntemi: {FACE_RECOGNITION_METHOD}")

# Employee modelini import etmeyi deniyorum
try:
    from employees.models import Employee
except ImportError:
    logger.warning("employees uygulaması veya Employee modeli bulunamadı. Raporlar isimsiz kaydedilebilir.")
    Employee = None

warnings.filterwarnings('ignore', category=UserWarning)

# Proje dizinlerini ayarlıyorum
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(STATIC_DIR, 'models')
logger.info(f"Proje Ana Dizini (BASE_DIR): {BASE_DIR}")
logger.info(f"Statik Dosya Dizini (STATIC_DIR): {STATIC_DIR}")
logger.info(f"Modeller Dizini (MODELS_DIR): {MODELS_DIR}")

# Kamera ayarlarını yapıyorum
CAMERA = {
    'index': 0,
    'width': 640,
    'height': 480
}

# Yüz tespiti için ayarlar
FACE_DETECTION = {
    'scale_factor': 1.3,
    'min_neighbors': 5,
    'min_size': (30, 30)
}

# Model ve dosya yollarını ayarlıyorum
MODEL_PATH = os.path.join(STATIC_DIR, "Yolo11n_50_epoch.pt")
NAMES_FILE = os.path.join(STATIC_DIR, 'names.json')
TRAINER_FILE = os.path.join(STATIC_DIR, 'trainer.yml')

# Haarcascade dosyasını bulmaya çalışıyorum
try:
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(CASCADE_PATH):
        logger.error(f"Haarcascade dosyası bulunamadı: {CASCADE_PATH}")
        CASCADE_PATH = os.path.join(STATIC_DIR, 'haarcascade_frontalface_default.xml')
        logger.warning(f"Statik klasördeki cascade kullanılacak: {CASCADE_PATH}")
except AttributeError:
    logger.warning("cv2.data.haarcascades bulunamadı. Cascade yolu manuel olarak ayarlanmalı.")
    CASCADE_PATH = os.path.join(STATIC_DIR, 'haarcascade_frontalface_default.xml')

# OpenCV DNN model dosyalarının yollarını ayarlıyorum
DNN_PROTOTXT_PATH = os.path.join(STATIC_DIR, 'deploy.prototxt.txt')
DNN_MODEL_PATH = os.path.join(STATIC_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# ArcFace model dosyalarının yollarını ayarlıyorum
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'ArcFaceResNet_epoch18_bs8_acc99.1.pth')
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, 'class_names.json')

# Raporlama gecikmesini ayarlıyorum (saniye)
REPORT_DELAY = 10

# --- ArcFace Model Tanımı (ResNet Tabanlı - BN ve Embedding ile) ---
class ArcFaceResNetModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, pretrained=False):
        super(ArcFaceResNetModel, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Son katmanı kaldırıyorum

        # Batch Normalization katmanını ekliyorum
        self.bn = nn.BatchNorm1d(in_features)

        # Embedding katmanını tanımlıyorum
        self.embedding_layer = nn.Linear(in_features, embedding_dim)

        # ArcFace sınıflandırma ağırlıklarını ayarlıyorum
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        features = self.backbone(x)
        features = self.bn(features)
        features = self.embedding_layer(features)
        features = nn.functional.normalize(features, p=2, dim=1)

        weight = nn.functional.normalize(self.weight, p=2, dim=1)
        cos = nn.functional.linear(features, weight)
        return cos

# --- Yüz Tanıma Sınıfı ---
class FaceRecognitionSystem:
    def __init__(self):
        self.method = FACE_RECOGNITION_METHOD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.names = {}
        self.cam = None
        self.dnn_face_detector = None

        # DNN yüz tespit modelini yüklüyorum
        if not os.path.exists(DNN_PROTOTXT_PATH) or not os.path.exists(DNN_MODEL_PATH):
            logger.error(f"DNN yüz tespit model dosyaları bulunamadı: Prototxt='{DNN_PROTOTXT_PATH}', Model='{DNN_MODEL_PATH}'")
        else:
            try:
                self.dnn_face_detector = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT_PATH, DNN_MODEL_PATH)
                logger.info("OpenCV DNN yüz tespit modeli başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"OpenCV DNN yüz tespit modeli yüklenirken hata: {e}")
                self.dnn_face_detector = None

        # Seçilen yönteme göre modeli yüklüyorum
        try:
            if self.method == 'lbph':
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.load_lbph_model()
                self.load_lbph_names()
            elif self.method == 'arcface':
                self.recognizer = None
                self.load_arcface_model()
                self.load_arcface_class_names()
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                logger.error(f"Geçersiz yüz tanıma yöntemi: {self.method}")
                self.model_loaded = False
        except Exception as e:
            logger.exception(f"{self.method} modeli yüklenirken hata oluştu")
            self.model_loaded = False

    

    def load_arcface_model(self):
        """ArcFace ResNet modelini (.pth) yükler ve sınıf sayısı uyumluluğunu kontrol eder."""
        self.model_loaded = False
        if not os.path.exists(ARCFACE_MODEL_PATH):
            logger.error(f"ArcFace model dosyası bulunamadı: {ARCFACE_MODEL_PATH}")
            return
        if not os.path.exists(CLASS_NAMES_PATH):
            logger.error(f"ArcFace sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
            return

        try:
            # 1. JSON'dan beklenen sınıf sayısını oku
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names_list = json.load(f)
            num_classes_from_json = len(class_names_list)
            if num_classes_from_json == 0:
                logger.error("Sınıf isimleri dosyası (class_names.json) boş.")
                return
            logger.info(f"class_names.json dosyasından {num_classes_from_json} sınıf ismi okundu.")

            # 2. Model checkpoint'ini yükle
            logger.info(f"'{ARCFACE_MODEL_PATH}' modeli yükleniyor...")
            # Checkpoint'in doğrudan state_dict olduğunu varsayıyoruz (yaygın durum)
            state_dict = torch.load(ARCFACE_MODEL_PATH, map_location=self.device)
            logger.info("Model state_dict dosyası başarıyla yüklendi.")

            # 3. Modeli Tanımla (Güncellenmiş ArcFaceResNetModel kullanarak)
            self.arcface_model = ArcFaceResNetModel(num_classes=num_classes_from_json, pretrained=False).to(self.device)
            logger.info(f"ArcFaceResNetModel {num_classes_from_json} sınıf ile tanımlandı.")

            # 4. State Dict'ten Modelin Sınıf Sayısını Kontrol Et (weight anahtarı üzerinden)
            weight_key = 'weight' # Model tanımımızdaki parametre adı
            num_classes_from_model = -1
            if weight_key in state_dict:
                model_weight_shape = state_dict[weight_key].shape
                if len(model_weight_shape) >= 1:
                    num_classes_from_model = model_weight_shape[0]
                else:
                    logger.error(f"Model state_dict'indeki '{weight_key}' parametresinin şekli geçersiz: {model_weight_shape}")
                    return
            else:
                logger.error(f"Model state_dict dosyasında ('{ARCFACE_MODEL_PATH}') beklenen sınıflandırma ağırlık anahtarı ('{weight_key}') bulunamadı.")
                logger.info(f"State_dict içindeki anahtarlar (ilk 10): {list(state_dict.keys())[:10]}...")
                return

            logger.info(f"Model state_dict'i ({ARCFACE_MODEL_PATH}) {num_classes_from_model} sınıf ile eğitilmiş görünüyor.")

            # 5. Sınıf Sayılarını Karşılaştır
            if num_classes_from_json != num_classes_from_model:
                logger.error(f"Sınıf sayısı uyuşmazlığı! Model ({ARCFACE_MODEL_PATH}) {num_classes_from_model} sınıf bekliyor, ancak class_names.json {num_classes_from_json} sınıf içeriyor. Model yüklenemedi.")
                return

            # 6. State Dict'i Modele Yükle (strict=True ile)
            logger.info("Sınıf sayıları uyumlu. State dict modele yükleniyor...")
            try:
                # 'module.' ön ekini kontrol et ve temizle (varsa)
                if all(key.startswith('module.') for key in state_dict.keys()):
                    logger.info("State dict anahtarlarında 'module.' ön eki algılandı, temizleniyor...")
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

                self.arcface_model.load_state_dict(state_dict, strict=True)
                self.arcface_model.eval()
                self.model_loaded = True
                logger.info(f"ArcFace ResNet modeli ({ARCFACE_MODEL_PATH}) {num_classes_from_json} sınıf ile başarıyla yüklendi.")
            except RuntimeError as e:
                logger.error(f"State dict yüklenirken hata (muhtemelen anahtar uyuşmazlığı): {e}")
                logger.info("İpucu: Model tanımı (ArcFaceResNetModel) ile kaydedilen modelin state_dict'indeki anahtarlar hala tam eşleşmiyor olabilir.")
                self.model_loaded = False

        except FileNotFoundError:
            logger.error(f"Model veya sınıf dosyası bulunamadı. Kontrol edin: {ARCFACE_MODEL_PATH}, {CLASS_NAMES_PATH}")
            self.model_loaded = False
        except json.JSONDecodeError:
            logger.error(f"Sınıf isimleri dosyası ({CLASS_NAMES_PATH}) geçerli bir JSON değil.")
            self.model_loaded = False
        except KeyError as e:
            logger.error(f"Model state_dict dosyasında beklenen anahtar bulunamadı: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.exception(f"ArcFace modeli yüklenirken beklenmedik bir hata oluştu")
            self.model_loaded = False

    def load_arcface_class_names(self):
        """ArcFace için index-isim eşleşmelerini JSON dosyasından yükler."""
        if not os.path.exists(CLASS_NAMES_PATH):
            logger.warning(f"ArcFace sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}. İsimler 'Unknown' olarak gösterilecek.")
            self.names = {}
            return
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                # Dosya ["isim1", "isim2", ...] formatında olmalı
                class_list = json.load(f)
                self.names = {i: name for i, name in enumerate(class_list)} # { index: name }
            logger.info(f"ArcFace sınıf isimleri ({CLASS_NAMES_PATH}) yüklendi: {list(self.names.values())}")
        except Exception as e:
            logger.error(f"ArcFace sınıf isimleri ({CLASS_NAMES_PATH}) yüklenirken hata: {e}")
            self.names = {}

    def release_camera(self):
        """Kamerayı serbest bırakır."""
        if self.cam is not None:
            if self.cam.isOpened():
                self.cam.release()
            self.cam = None
            logger.info("Kamera serbest bırakıldı.")

    def initialize_camera(self):
        """Kamerayı başlatır."""
        if self.cam is not None and self.cam.isOpened():
            logger.warning("Kamera zaten başlatılmış.")
            return True
        try:
            self.cam = cv2.VideoCapture(CAMERA['index'])
            if not self.cam.isOpened():
                logger.error(f"Webcam ({CAMERA['index']}) açılamadı.")
                self.cam = None
                return False

            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            actual_width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Kamera başarıyla başlatıldı. İstenen: {CAMERA['width']}x{CAMERA['height']}, Alınan: {int(actual_width)}x{int(actual_height)}")
            return True
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            if self.cam is not None:
                self.cam.release()
            self.cam = None
            return False

            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            actual_width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Kamera başarıyla başlatıldı. İstenen: {CAMERA['width']}x{CAMERA['height']}, Alınan: {int(actual_width)}x{int(actual_height)}")
            return True
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            if self.cam is not None:
                self.cam.release()
            self.cam = None
            return False

    def recognize_faces(self, img):
        """Verilen görüntüdeki yüzü, DNN kullanarak tespit eder ve seçilen yönteme göre tanır, yüz koordinatlarını döndürür."""
        name = "Unknown"
        confidence_score = 0
        recognized_face_coords_in_roi = None # Tespit edilen yüzün img içindeki (x,y,w,h) koordinatları

        # Yüz *tanıma* modeli yüklü mü kontrol et (ArcFace veya LBPH için)
        if self.method == 'arcface' and not self.model_loaded:
             logger.warning("ArcFace tanıma modeli yüklenmediği için yüz tanıma yapılamıyor.")
             return name, confidence_score, recognized_face_coords_in_roi
        # LBPH için model yükleme durumu kendi bloğunda ayrıca kontrol edilecek.

        # DNN yüz *tespit* modeli yüklü mü kontrol et
        if self.dnn_face_detector is None:
            logger.warning("DNN yüz tespit modeli yüklenmediği için yüz tespiti yapılamıyor.")
            return name, confidence_score, recognized_face_coords_in_roi

        try:
            if img is None or img.size == 0:
                logger.warning("recognize_faces fonksiyonuna gelen görüntü (img) boş veya geçersiz.")
                return "Error", 0, None

            (h_roi, w_roi) = img.shape[:2]
            if h_roi == 0 or w_roi == 0:
                logger.warning(f"recognize_faces içinde geçersiz ROI boyutları: {w_roi}x{h_roi}. Görüntü: {img.shape}")
                return "Error", 0, None

            # Giriş görüntüsünden (img, yani person_roi) blob oluştur
            # Model 300x300 BGR imaj bekler. Ortalama çıkarma değerleri (104.0, 117.0, 123.0) BGR içindir.
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 117.0, 123.0),
                                         swapRB=False, crop=False)

            self.dnn_face_detector.setInput(blob)
            detections = self.dnn_face_detector.forward() # shape: (1, 1, N, 7)

            best_detection_confidence = 0.0 # DNN tespit güveni için
            best_face_box = None # img (person_roi) içinde (x,y,w,h)

            # Tespitler üzerinde döngü
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2] # Tespitin güven skoru

                if confidence > 0.5: # DNN tespit güven eşiği
                    # Kutu koordinatlarını orijinal ROI boyutuna ölçekle
                    # detections[0,0,i,3:7] -> (startX, startY, endX, endY) normalize edilmiş (0-1 aralığında)
                    box_rel_x1 = detections[0, 0, i, 3]
                    box_rel_y1 = detections[0, 0, i, 4]
                    box_rel_x2 = detections[0, 0, i, 5]
                    box_rel_y2 = detections[0, 0, i, 6]

                    # Mutlak koordinatları img (person_roi) içinde hesapla
                    x = int(box_rel_x1 * w_roi)
                    y = int(box_rel_y1 * h_roi)
                    # DNN x2, y2 verir, bu yüzden width = (x2-x1)*w_roi, height = (y2-y1)*h_roi
                    w = int((box_rel_x2 - box_rel_x1) * w_roi)
                    h = int((box_rel_y2 - box_rel_y1) * h_roi)

                    # Pozitif genişlik ve yükseklik sağla, koordinatları sınırla
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)

                    if x + w > w_roi: w = w_roi - x # ROI sınırlarını aşmamasını sağla
                    if y + h > h_roi: h = h_roi - y # ROI sınırlarını aşmamasını sağla
                    
                    if w > 0 and h > 0: # Geçerli bir kutu mu?
                        if confidence > best_detection_confidence:
                            best_detection_confidence = confidence
                            best_face_box = (x, y, w, h)

            if best_face_box:
                x, y, w, h = best_face_box
                recognized_face_coords_in_roi = (x, y, w, h) # img (person_roi) içindeki koordinatlar
                
                # Yüz ROI'sini (BGR) çıkar
                face_roi_bgr = img[y:y+h, x:x+w]

                if face_roi_bgr.size == 0:
                    logger.warning("DNN ile tespit edilen yüz ROI'si (face_roi_bgr) boş.")
                    return name, confidence_score, recognized_face_coords_in_roi # Koordinatlar hala None olabilir

                # Şimdi seçilen yönteme göre yüzü TANIMA (ArcFace veya LBPH)
                if self.method == 'arcface':
                    # self.model_loaded zaten en başta kontrol edildi ArcFace için.
                    try:
                        face_pil = Image.fromarray(cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB))
                        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            outputs = self.arcface_model(face_tensor)
                        
                        probabilities = torch.softmax(outputs, dim=1)
                        current_confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 1)
                        
                        predicted_idx = predicted_idx_tensor.item()
                        current_confidence = current_confidence_tensor.item() * 100 # ArcFace güveni

                        if current_confidence > 40: # ArcFace tanıma güven eşiği
                            name = self.names.get(predicted_idx, "Unknown")
                            confidence_score = round(current_confidence)
                        else:
                            name = "Unknown"
                            confidence_score = round(current_confidence)
                            
                    except Exception as pred_err:
                        logger.exception(f"ArcFace predict sırasında hata oluştu (DNN tespit sonrası)")
                        name = "Error"
                
                elif self.method == 'lbph':
                    if self.recognizer is None or not self.model_loaded: 
                        logger.warning("LBPH modeli yüklenmedi veya recognizer None. Tanıma yapılamıyor.")
                        return name, confidence_score, recognized_face_coords_in_roi
                    try:
                        # LBPH gri tonlamalı resim bekler
                        if len(face_roi_bgr.shape) == 3 and face_roi_bgr.shape[2] == 3:
                             gray_face_for_lbph = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
                        elif len(face_roi_bgr.shape) == 2: # Zaten gri ise (normalde BGR olmalı)
                             gray_face_for_lbph = face_roi_bgr
                        else:
                             logger.warning(f"LBPH için beklenmedik yüz ROI formatı: {face_roi_bgr.shape}")
                             return "Error", 0, recognized_face_coords_in_roi

                        if gray_face_for_lbph.size == 0:
                            logger.warning("LBPH için gri tonlamalı yüz ROI'si boş.")
                            return "Error", 0, recognized_face_coords_in_roi
                            
                        label, conf = self.recognizer.predict(gray_face_for_lbph) # LBPH conf (uzaklık)
                        
                        # LBPH'de düşük conf daha iyi. % olarak göstermek için çevir.
                        # 100'lük bir eşik (trainer.yml'deki max uzaklık)
                        if conf < 100: 
                            name_candidate = self.names.get(str(label)) # LBPH isimleri {id_string: name}
                            if name_candidate:
                                name = name_candidate
                                confidence_score = round(100 - conf) # Yüzdeye çevir
                            else:
                                name = "Unknown" # Etiket names.json'da yok
                                confidence_score = round(100 - conf)
                        else:
                            name = "Unknown"
                            confidence_score = round(100 - conf) # Düşük güven skoru olacak
                    except cv2.error as cv2_lbph_err:
                        logger.exception(f"LBPH predict sırasında OpenCV hatası: {cv2_lbph_err}")
                        name = "Error"
                    except Exception as pred_err:
                        logger.exception(f"LBPH predict sırasında genel hata: {pred_err}")
                        name = "Error"
            # else: DNN ile yeterli güven skoruna sahip yüz tespit edilemedi, recognized_face_coords_in_roi None kalır

        except cv2.error as cv2_dnn_err: # cv2.dnn kaynaklı hataları yakala
            logger.error(f"OpenCV DNN (blob/forward) hatası (recognize_faces): {cv2_dnn_err}", exc_info=True)
            name = "Error" # recognized_face_coords_in_roi None kalır
        except Exception as e:
            logger.exception(f"Yüz tanıma (recognize_faces with DNN) genel hatası")
            name = "Error" # recognized_face_coords_in_roi None kalır

        return name, confidence_score, recognized_face_coords_in_roi # recognized_face_coords_in_roi, img içindeki koordinatlardır


# --- Frame İşleyici Sınıfı ---
class FrameProcessor:
    def __init__(self, frame_queue, result_queue, model, width=640):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model
        self.width = width
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_frames, name="FrameProcessorThread")
            self.thread.daemon = True
            self.thread.start()
            logger.info("Frame işleme thread'i başlatıldı.")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=1.0)
                if self.thread.is_alive():
                    logger.warning("Frame işleme thread'i düzgün kapanmadı.")

                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: break
                while not self.result_queue.empty():
                    try: self.result_queue.get_nowait()
                    except queue.Empty: break

                logger.info("Frame işleme thread'i durduruldu ve kuyruklar temizlendi.")
        self.thread = None

    def _process_frames(self):
        while self.running:
            try:
                original_frame = self.frame_queue.get(timeout=0.5)
                frame_resized = imutils.resize(original_frame, width=self.width)
                results = self.model.track(frame_resized, verbose=False, device=stay_safe_app.device)

                try:
                    self.result_queue.put((original_frame, results), timeout=0.5)
                except queue.Full:
                    logger.warning("Sonuç kuyruğu dolu. Bir sonuç atlanıyor.")
                    pass

            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"Frame işleme hatası: {e}", exc_info=True)
                try:
                    self.result_queue.put((original_frame, None), timeout=0.1)
                except queue.Full:
                    pass
                except NameError:
                    pass


# --- Ana Uygulama Sınıfı ---
class StaySafeApp:
    def __init__(self, model_path: str, width=640, height=480):
        self.model_path = model_path
        self.width = width
        self.height = height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Kullanılan cihaz: {self.device}")

        if not os.path.exists(self.model_path):
            logger.error(f"YOLO modeli bulunamadı: {self.model_path}")
            raise FileNotFoundError(f"Gerekli YOLO modeli bulunamadı: {self.model_path}")
        self.model = self.create_yolo_model()

        self.face_recognizer = FaceRecognitionSystem()

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        self.frame_processor = FrameProcessor(
            self.frame_queue,
            self.result_queue,
            self.model,
            self.width
        )

        self.camera_active = False
        self.predicted_names = []
        self.worker_info_cache = {}
        self.unsafe_persons_tracker = {}
        self.report_delay = REPORT_DELAY
        self._toggle_lock = threading.Lock()

    def create_yolo_model(self):
        """YOLO modelini yükler."""
        try:
            model = YOLO(self.model_path)
            model.to(self.device) # Modeli uygun cihaza taşı
            # İlk tahmini yaparak model ısıtılabilir (opsiyonel)
            # _ = model(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            logger.info("YOLO modeli başarıyla oluşturuldu ve cihaza yüklendi.")
            return model
        except Exception as e:
            logger.error(f"YOLO modeli oluşturma/yükleme hatası: {e}", exc_info=True)
            raise

    def findWorker(self, name):
        """Verilen isimdeki çalışanı Django veritabanında (Employee modeli) arar ve cache kullanır."""
        if not name or name in ["Unknown", "Error"]:
            return "Tanımsız veya hatalı çalışan adı."

        # Cache kontrolü
        if name in self.worker_info_cache:
            return self.worker_info_cache[name]

        # Employee modeli mevcut mu kontrol et (başlangıçta import edilmiş olmalı)
        if Employee is None:
            info = "Çalışan modeli (Employee) yüklenemediği için arama yapılamıyor."
            self.worker_info_cache[name] = info
            return info

        try:
            # Django ORM kullanarak çalışanı bul (isme göre, büyük/küçük harf duyarsız)
            worker = Employee.objects.filter(name__iexact=name.split()[0]).first() # İsim ve soyisim arasında boşluk varsa ilk kısmı al
            if worker:
                # worker bir Employee nesnesi
                # surname ve age alanlarının Employee modelinde olduğunu varsayıyoruz
                info = f"Çalışan: {worker.name} {getattr(worker, 'surname', '')} (ID: {worker.id}, Yaş: {getattr(worker, 'age', 'N/A')})"
                self.worker_info_cache[name] = info # Cache'e ekle
                return info
            else:
                 info = f"'{name}' isimli çalışan Django veritabanında bulunamadı."
                 self.worker_info_cache[name] = info # Bulunamadı bilgisini de cache'le
                 return info
        except Exception as e:
            logger.error(f"Çalışan arama hatası (Django DB - {name}): {e}")
            return f"'{name}' aranırken veritabanı hatası."

    def create_safety_report(self, person_id, recognized_name, frame_to_save, missing_equipment_list):
        """Veritabanına güvenlik ihlali raporu kaydeder ve eksik ekipmanları not eder."""
        logger.info(f"Rapor oluşturuluyor: ID={person_id}, İsim={recognized_name}, Eksik Ekipman={missing_equipment_list}")
        employee_instance = None
        if Employee and recognized_name not in ["Unknown", "Error", None]:
            try:
                # Django ORM kullanarak çalışan bulma
                
                employee_instance = Employee.objects.filter(name__iexact=recognized_name.split()[0]).first()
                if not employee_instance:
                     logger.warning(f"Rapor için çalışan bulunamadı (Django DB): {recognized_name}")
                # Alternatif: WorkerDatabase'den alınan ID ile Employee bulunur mu?
                # worker_db_info = self.database.find_employee(recognized_name)
                # if worker_db_info:
                #     employee_instance = Employee.objects.filter(id=worker_db_info[0]).first()

            except Exception as e:
                logger.error(f"Rapor için çalışan aranırken Django DB hatası: {e}")

        try:
            # Frame'i image dosyasına çevir
            ret, buffer = cv2.imencode('.jpg', frame_to_save)
            if not ret:
                logger.error("Rapor için görüntü JPEG formatına dönüştürülemedi.")
                return

            image_content = ContentFile(buffer.tobytes(), name=f'report_{person_id}_{int(time.time())}.jpg')

            # Eksik ekipman listesini metne çevir
            missing_equipment_str = ", ".join(missing_equipment_list) if missing_equipment_list else "Yok"

            # Rapor anındaki bilgileri almak için değişkenler
            pozisyon_adi = None
            vardiya_tipi = None
            supervizor_adi = None

            if employee_instance:
                try:
                    if employee_instance.pozisyon:
                        pozisyon_adi = employee_instance.pozisyon.pozisyon_ad
                    if employee_instance.vardiya:
                        vardiya_tipi = employee_instance.vardiya.get_vardiya_type_display()
                        if employee_instance.vardiya.vardiya_supervizor:
                            sup = employee_instance.vardiya.vardiya_supervizor
                            supervizor_adi = f"{sup.name} {sup.surname}"
                except Exception as info_err:
                    logger.warning(f"Rapor için ek çalışan bilgileri (pozisyon/vardiya/süpervizör) alınırken hata: {info_err}")

            # Raporu oluştur ve ek bilgileri ata
            report = EmployeeReport(
                employee=employee_instance,
                is_equipped=False,
                image=image_content,
                location="Kamera Görüntüsü",
                missing_equipment=missing_equipment_str,
                reported_pozisyon=pozisyon_adi,
                reported_vardiya=vardiya_tipi,
                reported_supervizor_name=supervizor_adi
            )
            report.save()
            logger.info(f"Güvenlik raporu başarıyla kaydedildi: ID={report.id}, Çalışan={employee_instance}")

            # Rapor oluşturulduktan sonra tracker'daki reported flag'ini güncellemek yerine
            # doğrudan silmek daha basit olabilir veya raporlandı olarak işaretlemek.
            # Bu işlem process_detection_results içinde yapılıyor.

        except Exception as e:
            logger.error(f"Güvenlik raporu kaydedilirken hata: {e}", exc_info=True)

    # --- process_detection_results için Yardımcı Metotlar ---

    def _get_persons_from_results(self, results):
        """YOLO sonuçlarından 'person' sınıfına ait kutuları çıkarır."""
        persons = []
        if results is not None and results[0].boxes is not None:
            class_names = self.model.names
            boxes = results[0].boxes.cpu().numpy()
            for box in boxes:
                try:
                    class_id = int(box.cls[0])
                    if class_id < len(class_names) and class_names[class_id] == 'person':
                        persons.append(box)
                except (IndexError, ValueError) as e:
                    logger.warning(f"Kutu işlenirken hata (cls): {e}, Kutu: {box}")
                    continue # Bu kutuyu atla
        return persons

    def _check_ppe_for_person(self, person_box, all_boxes, frame_shape):
        """Belirli bir kişi kutusu içindeki kask ve yeleği kontrol eder."""
        has_helmet = False
        has_vest = False
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        h_frame, w_frame = frame_shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x1 >= x2 or y1 >= y2: return has_helmet, has_vest # Geçersiz kutu

        class_names = self.model.names
        for other_box in all_boxes:
            # Kendisiyle karşılaştırmayı atla
            if np.array_equal(other_box.xyxy, person_box.xyxy): continue

            try:
                other_class_id = int(other_box.cls[0])
                if other_class_id >= len(class_names): continue
                other_class_name = class_names[other_class_id]

                if other_class_name in ['helmet', 'vest']:
                    ox1, oy1, ox2, oy2 = map(int, other_box.xyxy[0])
                    # Ekipmanın merkezi kişinin kutusu içinde mi?
                    center_x, center_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                    if x1 < center_x < x2 and y1 < center_y < y2:
                        if other_class_name == 'helmet': has_helmet = True
                        if other_class_name == 'vest': has_vest = True
                        # İkisi de bulunduysa döngüden çıkabiliriz (optimizasyon)
                        # if has_helmet and has_vest: break
            except (IndexError, ValueError) as e:
                logger.warning(f"Ekipman kontrolünde kutu işlenirken hata: {e}, Kutu: {other_box}")
                continue

        return has_helmet, has_vest

    def _update_unsafe_tracker(self, person_id, is_safe, recognized_name, person_roi):
        """Güvensiz kişi takipçisini günceller ve rapor gerekip gerekmediğini döndürür."""
        current_time = time.time()
        report_cooldown = 60 # Saniye
        should_create_report = False
        missing_equipment_list = [] # Rapor için eksik ekipman listesi

        if is_safe:
            # Güvenli hale geldiyse takipten çıkar
            if person_id in self.unsafe_persons_tracker:
                logger.debug(f"{person_id} güvenli hale geldi, takipten çıkarılıyor.")
                del self.unsafe_persons_tracker[person_id]
        else: # Güvensiz durum
            if person_id not in self.unsafe_persons_tracker:
                # İlk kez güvensiz görüldü, takibe al
                #logger.info(f"{person_id} güvensiz tespit edildi, takip başlatılıyor.")
                self.unsafe_persons_tracker[person_id] = {
                    'timestamp': current_time,
                    'reported': False,
                    'last_seen_frame': person_roi.copy() if person_roi is not None else None,
                    'last_report_time': 0
                }
            else:
                # Zaten takipte, süreyi ve rapor durumunu kontrol et
                tracker_entry = self.unsafe_persons_tracker[person_id]
                time_elapsed = current_time - tracker_entry['timestamp']
                can_report_again = current_time - tracker_entry.get('last_report_time', 0) > report_cooldown

                # Raporlama koşulları:
                # 1. Yeterli süre geçtiyse (report_delay)
                # 2. Henüz raporlanmadıysa VEYA raporlanmadı ama soğuma süresi bittiyse
                # 3. Yüz tanındıysa (Unknown/Error değilse)
                should_report_trigger = (time_elapsed >= self.report_delay and
                                         (not tracker_entry['reported'] or can_report_again) and
                                         recognized_name not in ["Unknown", "Error", None])

                if should_report_trigger:
                    logger.info(f"{person_id} ({recognized_name}) için raporlama koşulları sağlandı.")
                    should_create_report = True
                    # Rapor için son görülen frame'i güncelle (bu frame olabilir)
                    if person_roi is not None: tracker_entry['last_seen_frame'] = person_roi.copy()
                    tracker_entry['reported'] = True # Raporlanacak olarak işaretle (rapor başarılı olursa kalıcı)
                    tracker_entry['last_report_time'] = current_time
                    # Opsiyonel: timestamp'i sıfırlayıp delay'i tekrar başlatmak?
                    # tracker_entry['timestamp'] = current_time

                elif not tracker_entry['reported']:
                    # Raporlanmadıysa ve rapor süresi dolmadıysa, son frame'i güncelle
                     if person_roi is not None: tracker_entry['last_seen_frame'] = person_roi.copy()

        return should_create_report, missing_equipment_list # Şimdilik missing_equipment_list boş, bunu _check_ppe_for_person'dan alacağız

    def _draw_ppe_labels(self, frame, person_box, has_helmet, has_vest):
         """Tespit edilen veya eksik KKD etiketlerini frame üzerine çizer."""
         x1, y1, x2, y2 = map(int, person_box.xyxy[0])
         h_frame, w_frame = frame.shape[:2]
         label_font_scale = 0.5
         label_thickness = 1
         label_padding = 5

         missing_items = []
         equipped_items = []
         if not has_helmet: missing_items.append("BARET")
         else: equipped_items.append("BARET")
         if not has_vest: missing_items.append("YELEK")
         else: equipped_items.append("YELEK")

         # Başlangıç Y Offset'i (ana durum metninin altına gelmesi için yaklaşık)
         base_y_offset = 25 # Ana metin kutusunun yaklaşık yüksekliği

         # --- Eksik Ekipman Etiketlerini Çiz (Kırmızı, Sağ Taraf) ---
         current_y_offset_right = base_y_offset
         label_margin_x_right = 10
         for item_text in missing_items:
             (label_w, label_h), _ = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
             rect_x1 = x2 + label_margin_x_right
             rect_y1 = y1 + current_y_offset_right
             rect_x2 = rect_x1 + label_w + 2 * label_padding
             rect_y2 = rect_y1 + label_h + 2 * label_padding
             if rect_x2 < w_frame and rect_y2 < h_frame:
                 cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                 text_x = rect_x1 + label_padding
                 text_y_label = rect_y1 + label_h + label_padding
                 cv2.putText(frame, item_text, (text_x, text_y_label), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), label_thickness)
             current_y_offset_right += label_h + 2 * label_padding + 5

         # --- Takılı Ekipman Etiketlerini Çiz (Yeşil, Sol Taraf) ---
         current_y_offset_left = base_y_offset
         label_margin_x_left = 10
         for item_text in equipped_items:
             (label_w, label_h), _ = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
             rect_x2 = x1 - label_margin_x_left
             rect_y1 = y1 + current_y_offset_left
             rect_x1 = rect_x2 - label_w - 2 * label_padding
             rect_y2 = rect_y1 + label_h + 2 * label_padding
             if rect_x1 > 0 and rect_y2 < h_frame:
                 cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
                 text_x = rect_x1 + label_padding
                 text_y_label = rect_y1 + label_h + label_padding
                 cv2.putText(frame, item_text, (text_x, text_y_label), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), label_thickness)
             current_y_offset_left += label_h + 2 * label_padding + 5

    def _draw_person_status(self, frame, person_box, status_text, box_color):
         """Kişi durum metnini ve ana sınırlayıcı kutuyu çizer."""
         x1, y1, x2, y2 = map(int, person_box.xyxy[0])
         cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
         (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
         text_y_status = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 5
         cv2.rectangle(frame, (x1, text_y_status - text_height - baseline), (x1 + text_width, text_y_status + baseline), (0,0,0), -1)
         cv2.putText(frame, status_text, (x1, text_y_status), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    def _cleanup_stale_trackers(self, processed_person_ids):
        """Bu frame'de görülmeyen eski takipçileri temizler."""
        current_time = time.time()
        report_cooldown = 60
        stale_ids = set(self.unsafe_persons_tracker.keys()) - processed_person_ids
        for stale_id in stale_ids:
            tracker_entry = self.unsafe_persons_tracker[stale_id]
            time_since_first_seen = current_time - tracker_entry['timestamp'] # İlk görüldüğü zamandan beri geçen süre
            last_report_time = tracker_entry.get('last_report_time', 0)
            is_reported = tracker_entry.get('reported', False)
            cooldown_active = is_reported and (current_time - last_report_time <= report_cooldown)

            # Silme koşulları:
            # 1. Raporlanmadıysa ve uzun süre (delay*2) görülmediyse sil
            if not is_reported and time_since_first_seen > self.report_delay * 2:
                logger.debug(f"Takipteki {stale_id} uzun süredir görülmedi (raporlanmadı), takipten çıkarılıyor.")
                del self.unsafe_persons_tracker[stale_id]
            # 2. Raporlandıysa, soğuma süresi bittiyse VE uzun süre (delay*5) görülmediyse sil
            elif is_reported and not cooldown_active and time_since_first_seen > self.report_delay * 5:
                logger.debug(f"Raporlanan {stale_id} uzun süredir görülmedi (soğuma bitti), takipten çıkarılıyor.")
                del self.unsafe_persons_tracker[stale_id]

    # --- Ana İşleme Fonksiyonu (Refaktör Edildi) ---
    def process_detection_results(self, frame, results):
        """Tespit sonuçlarını işler, yüz tanıma yapar, raporlamayı yönetir ve frame üzerine çizer."""
        self.predicted_names = [] # Her frame için listeyi temizle
        current_time = time.time()
        processed_person_ids_in_frame = set() # Bu frame'de işlenen kişileri tutalım

        if results is None:
             cv2.putText(frame, "Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             return frame

        try:
            # 1. Tüm kutuları ve kişi kutularını al
            all_boxes = results[0].boxes.cpu().numpy() if results[0].boxes is not None else []
            person_boxes = self._get_persons_from_results(results)

            if not person_boxes:
                 self._cleanup_stale_trackers(processed_person_ids_in_frame) # Ekranda kimse yoksa bile eski takipçileri temizle
                 return frame

            # 2. Her kişiyi işle
            for person_box in person_boxes:
                x1, y1, x2, y2 = map(int, person_box.xyxy[0])
                h_frame, w_frame = frame.shape[:2]
                person_roi = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]

                if person_roi.size == 0: continue # Geçersiz ROI

                # 3. Ekipman Kontrolü
                has_helmet, has_vest = self._check_ppe_for_person(person_box, all_boxes, frame.shape)
                is_safe = has_helmet and has_vest

                # 4. Yüz Tespiti (her zaman) ve Yüz Tanıma (gerekirse)
                # recognize_faces her zaman çağrılır; yüz bulursa koordinatları, tanırsa ismi döndürür.
                # recognized_face_coords_in_roi eski isimdi, şimdi face_coords_for_blur kullanalım.
                potential_name, potential_confidence, face_coords_for_blur = \
                    self.face_recognizer.recognize_faces(person_roi)

                recognized_name_for_logic = "Unknown" # Raporlama ve takip için kullanılacak ana isim
                confidence_for_logic = 0

                if not is_safe:
                    # Güvensizse, ArcFace sonuçlarını mantık için kullan
                    recognized_name_for_logic = potential_name
                    confidence_for_logic = potential_confidence
                    self.predicted_names.append(recognized_name_for_logic) 
                else:
                    # Güvenliyse, ArcFace sonucunu mantık için kullanma ama listede belirt
                    if face_coords_for_blur: # Yüz tespit edildiyse (ArcFace sonucu ne olursa olsun)
                        # İsteğe bağlı: Güvenli ama tanınan bir yüz varsa bunu da loglayabilir veya predicted_names'e ekleyebiliriz.
                        # Örneğin: self.predicted_names.append(f"Safe ({potential_name})")
                        self.predicted_names.append("Safe (Face Detected)")
                    else:
                        self.predicted_names.append("Safe")

                # 5. Kişi ID'sini Belirle (recognized_name_for_logic'e göre)
                # Önceki ID mantığı devam edebilir, sadece recognized_name -> recognized_name_for_logic oldu
                if recognized_name_for_logic not in ["Unknown", "Error", None]:
                    person_id = recognized_name_for_logic
                else:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    person_id = f"person_at_{center_x}_{center_y}"
                    # Eğer yüz tespit edildi ama tanınmadıysa (güvenli veya güvensiz olabilir)
                    if face_coords_for_blur and (recognized_name_for_logic in ["Unknown", "Error", None]):
                        person_id += "_faced"

                processed_person_ids_in_frame.add(person_id)

                # 6. Takipçiyi Güncelle ve Raporlama İhtiyacını Kontrol Et
                # recognized_name_for_logic'i kullanarak rapor oluşturma kararı verilir.
                should_create_report, _ = self._update_unsafe_tracker(person_id, is_safe, recognized_name_for_logic, person_roi)

                # 7. Rapor Oluştur (Gerekliyse)
                if should_create_report:
                    missing_equipment_list_report = []
                    if not has_helmet: missing_equipment_list_report.append("Baret")
                    if not has_vest: missing_equipment_list_report.append("Yelek")

                    last_seen_frame_for_report = self.unsafe_persons_tracker[person_id].get('last_seen_frame')
                    if last_seen_frame_for_report is not None:
                         self.create_safety_report(person_id, recognized_name_for_logic, last_seen_frame_for_report, missing_equipment_list_report)
                    else:
                         logger.warning(f"{person_id} için rapor oluşturulamadı: Son frame bulunamadı.")
                         if person_id in self.unsafe_persons_tracker:
                            self.unsafe_persons_tracker[person_id]['reported'] = False
                            self.unsafe_persons_tracker[person_id]['last_report_time'] = 0

                # YÜZÜ BULANIKLAŞTIRMA (CANLI AKIŞTA) - Her zaman face_coords_for_blur kullanılır
                if face_coords_for_blur: # Eğer recognize_faces bir yüz koordinatı döndürdüyse
                    fx, fy, fw, fh = face_coords_for_blur
                    abs_face_x = x1 + fx 
                    abs_face_y = y1 + fy
                    abs_face_x_start = max(0, abs_face_x)
                    abs_face_y_start = max(0, abs_face_y)
                    abs_face_x_end = min(frame.shape[1], abs_face_x + fw)
                    abs_face_y_end = min(frame.shape[0], abs_face_y + fh)

                    if abs_face_x_start < abs_face_x_end and abs_face_y_start < abs_face_y_end:
                        face_region_to_blur = frame[abs_face_y_start:abs_face_y_end, abs_face_x_start:abs_face_x_end]
                        if face_region_to_blur.size > 0:
                            blurred_region = cv2.GaussianBlur(face_region_to_blur, (31, 31), 30)
                            frame[abs_face_y_start:abs_face_y_end, abs_face_x_start:abs_face_x_end] = blurred_region

                # 8. Çizim
                box_color = (0, 255, 0) if is_safe else (0, 0, 255)
                status_prefix = "Safe" if is_safe else "Unsafe"
                person_status_text = status_prefix
                
                # Çizim için gösterilecek isim ve durum:
                if not is_safe:
                    # Güvensiz durum: recognized_name_for_logic (ArcFace sonucu) kullanılır
                    if recognized_name_for_logic not in ["Unknown", "Error", None]:
                        person_status_text += f" ({recognized_name_for_logic} - {int(confidence_for_logic)}%)"
                    elif recognized_name_for_logic == "Error":
                        person_status_text += " (Face Rec Error)"
                    elif face_coords_for_blur: # Güvensiz, ArcFace tanıyamadı ama yüz var
                        person_status_text += " (Unknown Face)"
                    else: # Güvensiz, ArcFace tanıyamadı ve yüz de yok (cascade bulamadı)
                        person_status_text += " (No Face Detected)"
                else: # Güvenli durum
                    if face_coords_for_blur: # Güvenli ve yüz tespit edildi (ArcFace sonucu kullanılmıyor)
                        person_status_text += " (Face Detected)"
                    # else: Sadece "Safe" yazar (güvenli ve yüz tespit edilemedi)
                
                # Cooldown veya gecikme bilgisini ekle (sadece güvensizse ve takipteyse)
                if not is_safe and person_id in self.unsafe_persons_tracker:
                    tracker_entry = self.unsafe_persons_tracker[person_id]
                    time_elapsed = current_time - tracker_entry['timestamp']
                    if tracker_entry['reported']:
                        remaining_cooldown = int(60 - (current_time - tracker_entry.get('last_report_time', 0)))
                        if remaining_cooldown > 0:
                             person_status_text += f" (Raporlandi - {remaining_cooldown}s)"
                        # Raporlandı ve cooldown bittiyse ek bir şey yazmaya gerek yok, tekrar raporlanabilir.
                    elif time_elapsed > 1: # Henüz raporlanmadı ve gecikme süresi işliyor
                         person_status_text += f" ({int(time_elapsed)}s)"

                self._draw_person_status(frame, person_box, person_status_text, box_color)
                self._draw_ppe_labels(frame, person_box, has_helmet, has_vest)

            # 9. Eski Takipçileri Temizle
            self._cleanup_stale_trackers(processed_person_ids_in_frame)

        except Exception as e:
             logger.exception(f"Sonuç işleme hatası (process_detection_results)")
             cv2.putText(frame, "Result Processing Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return frame

    def toggle_camera(self):
        """Kamerayı ve ilgili işlemleri açıp kapatır."""
        with self._toggle_lock: # Kilidi al (Blok sonuna kadar tutar)
            try:
                if self.camera_active:
                    logger.info("Kamera kapatma işlemi başlıyor...")
                    logger.info("FrameProcessor durduruluyor...")
                    self.frame_processor.stop()
                    logger.info("Kamera serbest bırakılıyor...")
                    self.face_recognizer.release_camera()
                    self.camera_active = False
                    self.predicted_names = []
                    self.worker_info_cache = {}
                    self.unsafe_persons_tracker.clear()
                    logger.info("Kamera ve frame işleyici başarıyla durduruldu.")
                    return False
                else:
                    logger.info("Kamera başlatma işlemi başlıyor...")
                    logger.info("face_recognizer.initialize_camera() çağrılıyor...")
                    init_success = self.face_recognizer.initialize_camera()
                    if not init_success:
                        logger.error("Kamera başlatılamadı (initialize_camera başarısız).")
                        self.camera_active = False # Durumun False olduğundan emin ol
                        return False

                    logger.info("Kamera başarıyla başlatıldı. FrameProcessor başlatılıyor...")
                    self.frame_processor.start()
                    self.camera_active = True
                    logger.info("Kamera ve frame işleyici başarıyla başlatıldı.")
                    return True
            except Exception as e:
                logger.error(f"Kamera durumu değiştirilirken hata: {e}", exc_info=True)
                # Hata durumunda kaynakları serbest bırakmaya çalışalım
                try:
                    logger.warning("Hata nedeniyle kaynaklar temizleniyor...")
                    if hasattr(self, 'frame_processor') and self.frame_processor.running:
                        self.frame_processor.stop()
                    if hasattr(self, 'face_recognizer'):
                        self.face_recognizer.release_camera()
                except Exception as cleanup_err:
                    logger.error(f"Hata sonrası temizlik sırasında ek hata: {cleanup_err}")
                self.camera_active = False # Hata durumunda kapalı olarak işaretle
                self.unsafe_persons_tracker.clear()
                raise Exception(f"Kamera durumu değiştirilemedi: {str(e)}") # Hatayı view'a ilet


    def get_video_stream(self, recognition=False):
        """Video akışını üreten generator fonksiyonu."""
        if not self.camera_active or self.face_recognizer.cam is None or not self.face_recognizer.cam.isOpened():
            #logger.warning("Video akışı istendi ancak kamera aktif/açık değil.")
            # Kamera kapalıyken istemciye bilgi veren bir frame gönderelim
            error_frame = np.zeros((CAMERA['height'], CAMERA['width'], 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Off", (int(CAMERA['width']/2)-100, int(CAMERA['height']/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return # Akışı sonlandır

        logger.info("Video akış döngüsü başlıyor...")
        cap = self.face_recognizer.cam
        # YENİ LOGLAMA
        if cap is not None:
            logger.info(f"get_video_stream başlangıcında cap.isOpened(): {cap.isOpened()}")
        else:
            logger.error("get_video_stream başlangıcında cap (kamera nesnesi) None!")
            # Bu durumda zaten ilk baştaki `if not self.camera_active...` bloğu yakalamalı ama ek kontrol.
            # Kullanıcıya bilgi vermek için boş frame gönderelim ve çıkalım.
            error_frame = np.zeros((CAMERA['height'], CAMERA['width'], 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Obj. Error", (int(CAMERA['width']/2)-150, int(CAMERA['height']/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret_encode, buffer = cv2.imencode('.jpg', error_frame)
            if ret_encode:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        sent_camera_off_frame = False # Flag to send the off frame only once
        consecutive_read_failures = 0 # Yeni: Arka arkaya okuma hatası sayacı
        MAX_CONSECUTIVE_FAILURES = 30 # Yeni: Maksimum ardışık hata sayısı (yaklaşık 3 saniye @ 0.1s sleep)

        while self.camera_active: # Döngü kontrolü
            # 1. Kameradan Frame Oku
            ret, frame = cap.read()
            if not ret:
                consecutive_read_failures += 1
                logger.warning(f"Kameradan frame okunamadı ({consecutive_read_failures}/{MAX_CONSECUTIVE_FAILURES}). Sonraki frame deneniyor...")
                if consecutive_read_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(f"Kameradan {MAX_CONSECUTIVE_FAILURES} kez ardışık frame okunamadı. Akış sonlandırılıyor.")
                    self.camera_active = False # Kamerayı mantıksal olarak kapat
                    # İsteğe bağlı: toggle_camera çağrısı ile tam kapatma denenebilir ama lock mekanizmasına dikkat
                    break # Döngüyü sonlandır
                time.sleep(0.1) # Kısa bir süre bekle
                continue # Döngünün başına dön, sonraki frame'i dene
            
            consecutive_read_failures = 0 # Başarılı okuma, sayacı sıfırla
            frame = cv2.flip(frame, 1) # Görüntüyü aynala

            # 2. Frame'i İşleme Kuyruğuna Gönder
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                 try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                 except queue.Empty: pass
                 except queue.Full:
                     logger.warning("Frame kuyruğu anlık olarak çok dolu, frame atlanıyor.")
                     pass

            # 3. İşlenmiş Sonucu Al
            try:
                processed_frame, results = self.result_queue.get(timeout=0.5)

                # 4. Sonuçları İşle ve Görüntüyü Hazırla
                if recognition and processed_frame is not None:
                     output_frame = self.process_detection_results(processed_frame, results)
                elif processed_frame is not None:
                     output_frame = processed_frame
                else:
                     output_frame = frame
                     cv2.putText(output_frame, "Processing Issue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                # 5. Frame'i JPEG'e Çevir ve Gönder
                ret_encode, buffer = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret_encode:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    logger.warning("Frame JPEG formatına dönüştürülemedi.")

            except queue.Empty:
                time.sleep(0.02)
                continue
            # Bu blok güncellenecek
            except Exception as e:
                 logger.error(f"Video akışı ana döngü hatası (işleme/gönderme sırasında): {e}", exc_info=True)
                 # Hata oluştuğunda kamera hala açık mı kontrol et
                 if cap is not None and cap.isOpened():
                     logger.info("Ana döngüde hata oluştu ancak kamera (cap) hala açık görünüyor.")
                 elif cap is not None:
                     logger.warning("Ana döngüde hata oluştu ve kamera (cap) kapalı görünüyor.")
                 else:
                     logger.error("Ana döngüde hata oluştu ve kamera nesnesi (cap) None.")

                 # İstemciye bir hata frame'i göndermeyi dene
                 try:
                    error_frame = np.zeros((CAMERA['height'], CAMERA['width'], 3), dtype=np.uint8)
                    # Hata mesajını daha kısa tutalım, çok uzun olabilir
                    error_type_str = type(e).__name__
                    cv2.putText(error_frame, "Stream Processing Error", (int(CAMERA['width']/2)-200, int(CAMERA['height']/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.putText(error_frame, error_type_str, (10, CAMERA['height'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                    ret_encode, buffer = cv2.imencode('.jpg', error_frame)
                    if ret_encode:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                 except Exception as inner_e:
                     logger.error(f"Hata frame'i gönderilirken ek hata oluştu: {inner_e}")
                 
                 self.camera_active = False # Döngünün bir sonraki iterasyonda sonlanmasını garantile
                 # break # Döngü zaten self.camera_active = False ile sonlanacak

        # Döngü bitti (self.camera_active False oldu veya hata oluştu)
        logger.info("Video akış döngüsü sona erdi veya kamera kapatıldı.")

        # Son bir "Camera Off" frame'i gönder (eğer daha önce gönderilmediyse)
        if not sent_camera_off_frame:
            try:
                #logger.info("Kamera kapatıldığı/döngü bittiği için son 'Camera Off' frame gönderiliyor.")
                error_frame = np.zeros((CAMERA['height'], CAMERA['width'], 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Off", (int(CAMERA['width']/2)-100, int(CAMERA['height']/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret_encode, buffer = cv2.imencode('.jpg', error_frame)
                if ret_encode:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    sent_camera_off_frame = True # Gönderildi olarak işaretle
                else:
                     logger.warning("Kapanış frame'i JPEG formatına dönüştürülemedi.")
            except Exception as e:
                 logger.error(f"Kapanış frame'i gönderilirken hata: {e}")


# --- Global StaySafeApp Nesnesi ---
# Bu nesne Django uygulaması başlatıldığında bir kere oluşturulur.
stay_safe_app = None
try:
    logger.info("StaySafeApp uygulaması başlatılıyor...")
    stay_safe_app = StaySafeApp(model_path=MODEL_PATH)
    logger.info("StaySafeApp uygulaması başarıyla başlatıldı.")
except FileNotFoundError as e:
    logger.error(f"Uygulama başlatılamadı - Gerekli dosya bulunamadı: {e}", exc_info=True)
except Exception as e:
    logger.error(f"Uygulama başlatılırken kritik hata: {e}", exc_info=True)

# --- Dashboard Veri Yardımcı Fonksiyonları ---

def _get_dashboard_summary_data():
    """Dashboard özet verilerini hazırlıyorum."""
    today = timezone.now().date()
    total_reports = EmployeeReport.objects.count()
    today_reports = EmployeeReport.objects.filter(timestamp__date=today).count()
    return {
        'total_reports': total_reports,
        'today_reports': today_reports,
    }

def _get_daily_report_chart_data(days=7):
    """Son günlerin rapor sayılarını grafik için hazırlıyorum."""
    today = timezone.now().date()
    start_date = today - timedelta(days=days-1)
    reports_per_day = EmployeeReport.objects.filter(timestamp__date__gte=start_date)\
                                          .annotate(day=TruncDate('timestamp'))\
                                          .values('day')\
                                          .annotate(count=Count('id'))\
                                          .order_by('day')

    chart_labels = []
    chart_data = []
    report_counts_dict = {r['day']: r['count'] for r in reports_per_day}

    for i in range(days):
        current_day = start_date + timedelta(days=i)
        chart_labels.append(current_day.strftime('%d %b'))
        chart_data.append(report_counts_dict.get(current_day, 0))

    return json.dumps(chart_labels), json.dumps(chart_data)

def _get_employee_report_chart_data(top_n=5):
    """En çok raporlanan çalışanları grafik için hazırlıyorum."""
    employee_reports = EmployeeReport.objects.filter(employee__isnull=False)\
                                             .values('employee__name', 'employee__surname')\
                                             .annotate(
                                                 full_name=Concat(F('employee__name'), Value(' '), F('employee__surname')),
                                                 report_count=Count('id')
                                             )\
                                             .order_by('-report_count')[:top_n]

    chart_labels = [item['full_name'] for item in employee_reports]
    chart_data = [item['report_count'] for item in employee_reports]
    return json.dumps(chart_labels), json.dumps(chart_data)

def _get_missing_equipment_chart_data():
    """Eksik ekipman türlerine göre rapor sayılarını hesaplıyorum."""
    try:
        missing_equipment_counts = EmployeeReport.objects.exclude(missing_equipment__isnull=True).exclude(missing_equipment__exact='') \
                                        .values('missing_equipment') \
                                        .annotate(count=Count('id')) \
                                        .order_by('-count')

        labels = [item['missing_equipment'] for item in missing_equipment_counts]
        data = [item['count'] for item in missing_equipment_counts]

        if not labels:
            return [], []

        return labels, data
    except Exception as e:
        logger.error(f"Eksik ekipman grafik verisi alınırken hata: {e}")
        return [], []

# --- Django Views ---

def check_app_status(func):
    """Uygulama durumunu kontrol eden decorator."""
    def wrapper(request, *args, **kwargs):
        if stay_safe_app is None:
            logger.error(f"{func.__name__} çağrıldı ancak uygulama başlatılamamış.")
            if request.accepts('application/json'):
                return JsonResponse({'status': 'error', 'error': 'Uygulama düzgün başlatılamadı.'}, status=503)
            else:
                return render(request, 'display/error.html', {'error_message': 'Uygulama şu anda kullanılamıyor.'})
        return func(request, *args, **kwargs)
    return wrapper

def index(request):
    """Ana sayfa view'ı."""
    app_ready = stay_safe_app is not None
    camera_status = stay_safe_app.camera_active if app_ready else False
    context = {
        'camera_is_active': camera_status,
        'app_ready': app_ready,
        'error_message': None if app_ready else "Uygulama başlatılamadı. Lütfen kayıtları kontrol edin."
    }
    return render(request, 'display/index.html', context)

def home(request):
    """Dashboard sayfası view'ı."""
    app_ready = stay_safe_app is not None
    context = {
        'app_ready': app_ready,
        'error_message': None if app_ready else "Uygulama başlatılamadı. Lütfen kayıtları kontrol edin."
    }

    if not app_ready:
        return render(request, 'display/home.html', context)

    summary_data = _get_dashboard_summary_data()
    chart_labels, chart_data = _get_daily_report_chart_data()
    employee_chart_labels, employee_chart_data = _get_employee_report_chart_data()
    missing_equipment_labels, missing_equipment_data = _get_missing_equipment_chart_data()
    recent_reports = EmployeeReport.objects.all().order_by('-timestamp')[:5]

    context = {
        'app_ready': True,
        'total_reports': summary_data.get('total_reports', 0),
        'today_reports': summary_data.get('today_reports', 0),
        'camera_is_active': stay_safe_app.camera_active if app_ready else False,
        'chart_labels': chart_labels,
        'chart_data': chart_data,
        'employee_chart_labels': employee_chart_labels,
        'employee_chart_data': employee_chart_data,
        'missing_equipment_labels': missing_equipment_labels,
        'missing_equipment_data': missing_equipment_data,
        'recent_reports': recent_reports,
    }

    return render(request, 'display/home.html', context)

@check_app_status
def video_feed(request):
    """Video akışı view'ı."""
    if not stay_safe_app.camera_active:
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame")

    return StreamingHttpResponse(stay_safe_app.get_video_stream(recognition=True),
                                 content_type="multipart/x-mixed-replace; boundary=frame")

@csrf_exempt
@check_app_status
def toggle_camera(request):
    """Kamera durumunu değiştiren view."""
    if request.method == 'POST':
        try:
            is_active = stay_safe_app.toggle_camera()
            return JsonResponse({
                'status': 'success',
                'camera_is_active': is_active
            })
        except Exception as e:
            logger.error(f"Kamera durumu değiştirilirken hata: {e}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'error': f"Kamera durumu değiştirilemedi: {str(e)}"
            }, status=500)
    else:
        return JsonResponse({'status': 'error', 'error': 'Geçersiz istek metodu'}, status=405)

@csrf_exempt
@check_app_status
def get_worker_info(request):
    """Çalışan bilgilerini döndüren view."""
    if request.method == 'GET':
        if not stay_safe_app.camera_active:
            return JsonResponse({'status': 'info', 'message': 'Kamera kapalı.'})

        last_valid_name = None
        for name in reversed(stay_safe_app.predicted_names):
            if name and name not in ["Unknown", "Error"]:
                last_valid_name = name
                break

        if last_valid_name:
            worker_info = stay_safe_app.findWorker(last_valid_name)
            return JsonResponse({'status': 'success', 'worker_info': worker_info})
        else:
            last_prediction = stay_safe_app.predicted_names[-1] if stay_safe_app.predicted_names else "None"
            return JsonResponse({'status': 'info', 'message': f'Geçerli çalışan tanınmadı. Son tahmin: {last_prediction}'})
    else:
        return JsonResponse({'status': 'error', 'error': 'Geçersiz istek metodu'}, status=405)

def report_list(request):
    """Rapor listesi view'ı."""
    reports = EmployeeReport.objects.all()
    context = {
        'reports': reports
    }
    return render(request, 'reports/report_list.html', context)