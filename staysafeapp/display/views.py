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
# import sqlite3 # Kaldırıldı
import warnings
# from imutils.video import FPS # FPS kullanılmıyor gibi, kaldırılabilir
from ultralytics import YOLO
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile # Rapor görüntüsü kaydetmek için
from django.core.files.storage import default_storage # Dosya işlemleri için
from reports.models import EmployeeReport # Rapor modelini import et
from django.utils import timezone # Zaman dilimi için
from datetime import timedelta # Zaman farkı hesaplamak için
from django.db.models.functions import TruncDate, Concat # Tarihe göre gruplamak için, Concat eklendi
from django.db.models import Count, F, Value # F ve Value eklendi

# ArcFace için eklenenler
import torch.nn as nn
from torchvision.transforms import v2 as T
from PIL import Image

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Yüz Tanıma Yöntemi Seçimi ---
# Değerler: 'lbph' veya 'arcface'
#FACE_RECOGNITION_METHOD = input('Yüz tanıma yöntemi seçin (lbph/arcface): ')
FACE_RECOGNITION_METHOD = 'arcface'
logger.info(f"Kullanılacak yüz tanıma yöntemi: {FACE_RECOGNITION_METHOD}")


try:
    from employees.models import Employee
except ImportError:
    logger.warning("employees uygulaması veya Employee modeli bulunamadı. Raporlar isimsiz kaydedilebilir.")
    Employee = None

warnings.filterwarnings('ignore', category=UserWarning)

# --- Ayarlar ve Sabitler ---
# Projenin ana dizinini (manage.py'nin olduğu yer)
# Bu yapıya göre BASE_DIR, 'staysafeapp' klasörü
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(STATIC_DIR, 'models') # ArcFace modelinin kaydedildiği yer
logger.info(f"Proje Ana Dizini (BASE_DIR): {BASE_DIR}")
logger.info(f"Statik Dosya Dizini (STATIC_DIR): {STATIC_DIR}")
logger.info(f"Modeller Dizini (MODELS_DIR): {MODELS_DIR}")


# Kamera ayarları
CAMERA = {
    'index': 0,  # Default camera
    'width': 640,
    'height': 480
}

# Yüz tanıma ayarları
FACE_DETECTION = {
    'scale_factor': 1.3,
    'min_neighbors': 5,
    'min_size': (30, 30)
}

# Dosya yolları (STATIC_DIR kullanarak)
MODEL_PATH = os.path.join(STATIC_DIR, "Yolo11n_50_epoch.pt")
# DB_PATH = os.path.join(STATIC_DIR, "Workers.db") # Kaldırıldı
NAMES_FILE = os.path.join(STATIC_DIR, 'names.json')
TRAINER_FILE = os.path.join(STATIC_DIR, 'trainer.yml')
# Haarcascade dosyasını OpenCV'nin kurulu olduğu yerden alındı
try:
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(CASCADE_PATH):
        logger.error(f"Haarcascade dosyası bulunamadı: {CASCADE_PATH}")
        CASCADE_PATH = os.path.join(STATIC_DIR, 'haarcascade_frontalface_default.xml') # Statik klasörde de arandı
        logger.warning(f"Statik klasördeki cascade kullanılacak: {CASCADE_PATH}")

except AttributeError:
    logger.warning("cv2.data.haarcascades bulunamadı. Cascade yolu manuel olarak ayarlanmalı veya static klasörüne konulmalı.")
    CASCADE_PATH = os.path.join(STATIC_DIR, 'haarcascade_frontalface_default.xml') # Statik klasördeki cascade kullanılacak

# ArcFace için eklenen dosya yolları
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'arcface_model.pth')
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, 'class_names.json')

REPORT_DELAY = 5 # Raporlama öncesi bekleme süresi (saniye)

# --- ArcFace Model Tanımı (faceRecognition_train/views.py'den alındı) ---
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(ArcFaceModel, self).__init__()
        # Basitleştirilmiş bir backbone (örnek, gerçek bir ResNet vb. kullanılabilir)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, embedding_dim)
        )
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # ArcFace loss katmanı burada tanımlanmıyor, sadece embedding çıkarılıyor
        # Sınıflandırma için bir lineer katman eklenebilir veya embedding karşılaştırması yapılabilir.
        # Şimdilik eğitimdeki yapıldı. Duruma göre değiştirecem.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        features = self.backbone(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        # Eğer sadece embedding isteniyorsa burayı döndür:
        # return features

        # Eğitimdeki gibi sınıflandırma çıktısı (logitler/cosine similarity):
        weight = nn.functional.normalize(self.weight, p=2, dim=1)
        cos = nn.functional.linear(features, weight)
        return cos

# --- Yüz Tanıma Sınıfı ---
class FaceRecognitionSystem:
    def __init__(self):
        self.method = FACE_RECOGNITION_METHOD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.names = {} # LBPH için {id: name}, ArcFace için {index: name}
        self.cam = None

        # Ortak: Yüz tespiti için cascade
        if not os.path.exists(CASCADE_PATH):
             logger.error(f"Yüz tespiti için cascade dosyası yüklenemedi: {CASCADE_PATH}")
             self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        # Modele özel yüklemeler
        try:
            if self.method == 'lbph':
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.load_lbph_model()
                self.load_lbph_names()
            elif self.method == 'arcface':
                self.recognizer = None # ArcFace için OpenCV recognizer kullanılmıyor
                self.load_arcface_model()
                self.load_arcface_class_names() # Class index -> name
                # ArcFace için gerekli dönüşümler
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

    def load_lbph_model(self):
        """LBPH modelini (.yml) yükler."""
        if not os.path.exists(TRAINER_FILE):
            logger.warning(f"LBPH Trainer dosyası bulunamadı: {TRAINER_FILE}. Yüz tanıma yapılamayacak.")
            return
        try:
            self.recognizer.read(TRAINER_FILE)
            self.model_loaded = True
            logger.info("LBPH yüz tanıma modeli (trainer.yml) başarıyla yüklendi.")
        except cv2.error as e:
             logger.error(f"LBPH Trainer dosyası ({TRAINER_FILE}) okunurken OpenCV hatası: {e}. Dosya bozuk veya uyumsuz olabilir.")
        except Exception as e:
            logger.error(f"LBPH Trainer dosyası ({TRAINER_FILE}) yüklenirken beklenmedik hata: {e}")

    def load_lbph_names(self):
        """LBPH için ID-İsim eşleşmelerini JSON dosyasından yükler."""
        if not os.path.exists(NAMES_FILE):
            logger.warning(f"LBPH Names dosyası bulunamadı: {NAMES_FILE}. İsimler 'Unknown' olarak gösterilecek.")
            self.names = {}
            return
        try:
            with open(NAMES_FILE, 'r', encoding='utf-8') as fs:
                content = fs.read().strip()
                if content:
                    self.names = json.loads(content) # { "id_str": "name" }
                    logger.info(f"LBPH İsim eşleşmeleri ({NAMES_FILE}) yüklendi: {self.names}")
                else:
                    logger.warning(f"LBPH Names dosyası ({NAMES_FILE}) boş.")
                    self.names = {}
        except Exception as e:
            logger.error(f"LBPH Names dosyası ({NAMES_FILE}) yüklenirken hata: {e}")
            self.names = {}

    def load_arcface_model(self):
        """ArcFace modelini (.pth) yükler."""
        if not os.path.exists(ARCFACE_MODEL_PATH):
            logger.error(f"ArcFace model dosyası bulunamadı: {ARCFACE_MODEL_PATH}")
            return
        if not os.path.exists(CLASS_NAMES_PATH):
            logger.error(f"ArcFace sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
            return # Model yüklense bile isimler olmadan anlamsız

        try:
            # Önce sınıf sayısını öğrenmemiz lazım
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            num_classes = len(class_names)
            if num_classes == 0:
                logger.error("Sınıf isimleri dosyası boş.")
                return

            # Modeli tanımla ve state dict'i yükle
            self.arcface_model = ArcFaceModel(num_classes=num_classes).to(self.device)
            self.arcface_model.load_state_dict(torch.load(ARCFACE_MODEL_PATH, map_location=self.device))
            self.arcface_model.eval() # Modeli çıkarım moduna al
            self.model_loaded = True
            logger.info(f"ArcFace modeli ({ARCFACE_MODEL_PATH}) ve {num_classes} sınıf başarıyla yüklendi.")
        except Exception as e:
            logger.exception(f"ArcFace modeli yüklenirken hata oluştu")
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

    def recognize_faces(self, img):
        """Verilen görüntüdeki yüzü, seçilen yönteme göre tanır."""
        name = "Unknown"
        confidence_score = 0

        if not self.model_loaded:
             return name, confidence_score # Model yoksa direkt Unknown dön

        if self.face_cascade is None:
            logger.warning("Face cascade yüklenmediği için yüz tespiti yapılamıyor.")
            return name, confidence_score

        try:
            # 1. Yüz Tespiti (Her iki yöntem için ortak)
            if len(img.shape) == 3 and img.shape[2] == 3:
                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                 gray = img
            else:
                 logger.warning(f"Beklenmedik görüntü formatı: shape={img.shape}")
                 return "Error", 0

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                face_roi_bgr = img[y:y+h, x:x+w] # Renkli ROI alalım (ArcFace için lazım)
                face_roi_gray = gray[y:y+h, x:x+w]

                if face_roi_bgr.size == 0 or face_roi_gray.size == 0:
                    logger.warning("Tespit edilen yüz ROI'si boş.")
                    return name, confidence_score

                # 2. Tanıma (Yönteme göre farklılık gösterir)
                if self.method == 'lbph':
                    try:
                        id_recognized, confidence = self.recognizer.predict(face_roi_gray)
                        if confidence < 80: # LBPH: Düşük confidence iyi
                            name = self.names.get(str(id_recognized), "Unknown")
                            confidence_score = round((1 - (confidence / 100)) * 100)
                        else:
                             name = "Unknown"
                             confidence_score = round((1 - (confidence / 100)) * 100)
                    except cv2.error as cv_err:
                        logger.warning(f"LBPH predict hatası: {cv_err}")
                        name = "Error"
                    except Exception as pred_err:
                        logger.error(f"LBPH predict sırasında beklenmedik hata: {pred_err}")
                        name = "Error"

                elif self.method == 'arcface':
                    try:
                        # Görüntüyü hazırla (PIL -> Transform -> Tensor)
                        face_pil = Image.fromarray(cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB))
                        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device) # Batch dim ekle

                        # Model çıkarımı
                        with torch.no_grad():
                            outputs = self.arcface_model(face_tensor)
                        
                        # Sonucu işle (Cosine similarity varsayımıyla)
                        probabilities = torch.softmax(outputs, dim=1) # Veya sadece argmax(outputs)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        predicted_idx = predicted_idx.item()
                        confidence = confidence.item() * 100 # Yüzdeye çevir

                        # Güven eşiği (ArcFace için % olarak)
                        if confidence > 60: # Eşik değeri (ayarlanabilir)
                            name = self.names.get(predicted_idx, "Unknown")
                            confidence_score = round(confidence)
                        else:
                            name = "Unknown"
                            confidence_score = round(confidence)
                            
                    except Exception as pred_err:
                        logger.exception(f"ArcFace predict sırasında hata oluştu")
                        name = "Error"
            # else: # Yüz tespit edilemedi
                # pass

        except Exception as e:
            logger.exception(f"Yüz tanıma genel hatası")
            name = "Error"

        return name, confidence_score


# --- Frame İşleyici Sınıfı ---
class FrameProcessor:
    def __init__(self, frame_queue, result_queue, model, width=640):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model
        self.width = width
        self.running = False
        self.thread = None
        # self.last_result = None # Cache mekanizması kaldırıldı
        # self.last_result_time = 0
        # self.result_cache_time = 0.1

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_frames, name="FrameProcessorThread")
            self.thread.daemon = True
            self.thread.start()
            logger.info("FrameProcessor iş parçacığı başlatıldı.")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread is not None:
                # Kuyrukları temizlemeden önce thread'in bitmesi bekleniyor
                self.thread.join(timeout=1.0) # 1 saniye bekle
                if self.thread.is_alive():
                    logger.warning("FrameProcessor thread'i join ile durmadı.")
                    # Burada belki daha zorlayıcı bir durdurma mekanizması gerekebilir
                    # ancak daemon thread olduğu için ana program bitince kapanacaktır.

                # Thread durduktan sonra kuyrukları temizle
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: break
                while not self.result_queue.empty():
                    try: self.result_queue.get_nowait()
                    except queue.Empty: break

                logger.info("FrameProcessor iş parçacığı durduruldu ve kuyruklar temizlendi.")
        self.thread = None


    def _process_frames(self):
        while self.running:
            try:
                # Timeout ile frame al, bloklamayı azaltır
                original_frame = self.frame_queue.get(timeout=0.5)

                # Frame'i yeniden boyutlandır (YOLO için)
                frame_resized = imutils.resize(original_frame, width=self.width)

                # YOLO modelini çalıştır
                # stream=True daha verimli olabilir ama sonuçları yönetmek farklılaşır
                results = self.model(frame_resized, verbose=False, device=stay_safe_app.device) # Cihazı belirt

                # Sonucu orijinal frame ile birlikte kuyruğa koy
                # Burası kritik, eğer result_queue doluysa takılma olabilir
                try:
                    self.result_queue.put((original_frame, results), timeout=0.5)
                except queue.Full:
                    logger.warning("Sonuç kuyruğu (result_queue) dolu. Bir sonuç atlanıyor.")
                    # Eski sonucu atıp yenisini ekleyebiliriz ama senkronizasyon bozulabilir
                    # Şimdilik sadece uyarı verilecek
                    pass

            except queue.Empty:
                # Frame kuyruğu boşsa kısa bir süre bekle
                time.sleep(0.01)
                continue
            except Exception as e:
                # Hata durumunda logla ve devam etmeye çalış
                logger.error(f"Frame işleme hatası (_process_frames): {e}", exc_info=True) # Hata detayını logla
                # Hata durumunda result_queue'ya None gönderilebilir.
                try:
                     # Orijinal frame'i None result ile gönderelim ki akış devam etsin
                    self.result_queue.put((original_frame, None), timeout=0.1)
                except queue.Full:
                     pass # Doluysa yapacak bir şey yok
                except NameError: # original_frame tanımlanmadan hata oluşmuşsa
                     pass


# --- Ana Uygulama Sınıfı ---
class StaySafeApp:
    def __init__(self, model_path: str, width=640, height=480):
        self.model_path = model_path
        self.width = width # YOLO'nun işleyeceği genişlik
        self.height = height # Kamera yüksekliği (bilgi amaçlı)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Kullanılan cihaz: {self.device}")

        if not os.path.exists(self.model_path):
            logger.error(f"YOLO modeli bulunamadı: {self.model_path}")
            raise FileNotFoundError(f"Gerekli YOLO modeli bulunamadı: {self.model_path}")
        self.model = self.create_yolo_model()

        self.face_recognizer = FaceRecognitionSystem()

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.deterministic = False # Genellikle False daha hızlıdır

        # Kuyruk boyutlarını artırılabilir. Makine hızına bağlı. Test edilecek.
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        self.frame_processor = FrameProcessor(
            self.frame_queue,
            self.result_queue,
            self.model,
            self.width # FrameProcessor'a işlenecek genişliği ver
        )

        self.camera_active = False # Başlangıçta kamera kapalı
        self.predicted_names = [] # Son frame'de tanınan isimleri tutar
        self.worker_info_cache = {} # Çalışan bilgilerini cache'le
        self.unsafe_persons_tracker = {} # Ekipmansız kişileri takip etmek için {person_id: {'timestamp': float, 'reported': bool, 'last_seen_frame': np.ndarray}}
        self.report_delay = REPORT_DELAY # Raporlama gecikmesi
        self._toggle_lock = threading.Lock() # Toggle işlemi için kilit


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
            worker = Employee.objects.filter(name__iexact=name).first()
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
                
                employee_instance = Employee.objects.filter(name__iexact=recognized_name).first()
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


    def process_detection_results(self, frame, results):
        """Tespit sonuçlarını işler, yüz tanıma yapar, raporlamayı yönetir ve frame üzerine çizer."""
        self.predicted_names = [] # Her frame için listeyi temizle
        current_time = time.time()
        processed_person_ids = set() # Bu frame'de işlenen kişileri tutalım
        report_cooldown = 60 # Saniye cinsinden raporlama soğuma süresi

        if results is None:
             cv2.putText(frame, "Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             # Tracker'ı burada temizlemeyelim, belki geçici bir hatadır.
             return frame

        try:
            class_names = self.model.names
            boxes = results[0].boxes.cpu().numpy()
            persons = [box for box in boxes if class_names[int(box.cls[0])] == 'person']

            if not persons:
                 # Ekranda kimse yoksa, tüm 'unsafe' takipçilerini temizleyebiliriz (opsiyonel)
                 # self.unsafe_persons_tracker.clear()
                 return frame

            # -- Sadece en büyük kişiyi değil, tüm kişileri işleyelim --
            for person_box in persons:
                x1, y1, x2, y2 = map(int, person_box.xyxy[0])
                h_frame, w_frame = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                if x1 >= x2 or y1 >= y2: continue # Geçersiz kutu

                has_helmet = False
                has_vest = False
                # Kask ve yelek kontrolü (bu person kutusu içinde)
                for other_box in boxes:
                    if np.array_equal(other_box.xyxy, person_box.xyxy): continue
                    other_class_id = int(other_box.cls[0])
                    if other_class_id >= len(class_names): continue
                    other_class_name = class_names[other_class_id]

                    if other_class_name in ['helmet', 'vest']:
                        ox1, oy1, ox2, oy2 = map(int, other_box.xyxy[0])
                        center_x, center_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                        if x1 < center_x < x2 and y1 < center_y < y2:
                            if other_class_name == 'helmet': has_helmet = True
                            if other_class_name == 'vest': has_vest = True
                            # if has_helmet and has_vest: break # Optimizasyon

                is_safe = has_helmet and has_vest
                box_color = (0, 255, 0) if is_safe else (0, 0, 255)
                status_prefix = "Safe" if is_safe else "Unsafe"
                person_status_text = status_prefix
                recognized_name = "Unknown"
                confidence = 0

                # Yüz tanımayı sadece güvensizse VEYA her zaman yapabiliriz?
                # Şimdilik sadece güvensizse yapalım.
                person_roi = frame[y1:y2, x1:x2]
                if not is_safe and person_roi.size > 0:
                    recognized_name, confidence = self.face_recognizer.recognize_faces(person_roi)
                    self.predicted_names.append(recognized_name) # Tahmin listesine ekle
                    if recognized_name not in ["Unknown", "Error"]:
                        person_status_text += f" ({recognized_name} - {confidence}%)"
                    elif recognized_name == "Error":
                        person_status_text += " (Face Rec Error)"
                    else: # Unknown
                        person_status_text += " (Unknown)"
                elif person_roi.size == 0:
                     person_status_text += " (ROI Error)"
                     # recognized_name = "Unknown" # Zaten varsayılan Unknown
                     self.predicted_names.append("Unknown")
                elif is_safe:
                    # Güvenli ise de tanıma yapıp ismi ekleyebiliriz (opsiyonel)
                    # recognized_name, confidence = self.face_recognizer.recognize_faces(person_roi)
                    pass # Güvenli ise status yeterli

                # --- Raporlama Mantığı --- 
                # Takip için benzersiz bir ID belirleyelim
                # Tanınan isim varsa onu, yoksa kutu merkezini kullanalım (basit yaklaşım)
                if recognized_name not in ["Unknown", "Error", None]:
                    person_id = recognized_name
                else:
                    # Tanınmayanlar için yaklaşık konum bazlı ID
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    person_id = f"unknown_at_{center_x}_{center_y}" # Bu ID çok stabil olmayabilir

                processed_person_ids.add(person_id) # Bu frame'de görüldü

                if is_safe:
                    # Eğer kişi güvenli hale geldiyse ve takip ediliyorsa, takipten çıkar
                    if person_id in self.unsafe_persons_tracker:
                        logger.debug(f"{person_id} güvenli hale geldi, takipten çıkarılıyor.")
                        del self.unsafe_persons_tracker[person_id]
                else: # Güvensiz durum
                    if person_id not in self.unsafe_persons_tracker:
                        # Kişi ilk kez güvensiz görüldü, takibe al
                        logger.info(f"{person_id} güvensiz tespit edildi, takip başlatılıyor.")
                        self.unsafe_persons_tracker[person_id] = {
                            'timestamp': current_time,
                            'reported': False,
                            'last_seen_frame': person_roi.copy(), # Rapor için ROI'yi sakla
                            'last_report_time': 0 # İlk başta raporlanmadı
                        }
                    else:
                        # Kişi zaten takipte, süreyi ve rapor durumunu kontrol et
                        tracker_entry = self.unsafe_persons_tracker[person_id]
                        time_elapsed = current_time - tracker_entry['timestamp']
                        can_report_again = current_time - tracker_entry.get('last_report_time', 0) > report_cooldown

                        # Süre dolduysa VE henüz raporlanmadıysa VEYA raporlandı ama soğuma süresi bittiyse VE yüz tanındıysa rapor oluştur
                        should_report = (time_elapsed >= self.report_delay and
                                         (not tracker_entry['reported'] or can_report_again) and
                                         recognized_name not in ["Unknown", "Error", None])

                        if should_report:
                            logger.info(f"{person_id} ({recognized_name}) için raporlama koşulları sağlandı. Rapor oluşturuluyor...")

                            # Eksik ekipmanları belirle
                            missing_equipment_list = []
                            if not has_helmet: missing_equipment_list.append("Baret")
                            if not has_vest: missing_equipment_list.append("Yelek")

                            # Rapor için son görülen frame'i (ROI) kullan
                            self.create_safety_report(person_id, recognized_name, tracker_entry['last_seen_frame'], missing_equipment_list)

                            # Raporlandı olarak işaretle ve son rapor zamanını güncelle
                            tracker_entry['reported'] = True
                            tracker_entry['last_report_time'] = current_time
                            # Opsiyonel: timestamp'i sıfırlayıp delay'i tekrar başlatabiliriz ama cooldown varken gerekmeyebilir
                            # tracker_entry['timestamp'] = current_time

                        elif not tracker_entry['reported']:
                             # Raporlanmadıysa son frame'i güncelle (henüz rapor süresi dolmamış olabilir)
                            self.unsafe_persons_tracker[person_id]['last_seen_frame'] = person_roi.copy()

                        # Sürekli güvensiz kalanlar için status'a uyarı ekleyebiliriz
                        if tracker_entry['reported'] and not can_report_again:
                             remaining_cooldown = int(report_cooldown - (current_time - tracker_entry['last_report_time']))
                             person_status_text += f" (Raporlandi - {remaining_cooldown}s)"
                        elif tracker_entry['reported']:
                             person_status_text += " (Raporlandi)" # Soğuma bitti ama hala güvensizse
                        elif time_elapsed > 1:
                            person_status_text += f" ({int(time_elapsed)}s)" # Kaç sn güvensiz

                # --- Çizim ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                (text_width, text_height), baseline = cv2.getTextSize(person_status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y_status = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 5
                # Ana durum metni için arka plan
                cv2.rectangle(frame, (x1, text_y_status - text_height - baseline), (x1 + text_width, text_y_status + baseline), (0,0,0), -1)
                cv2.putText(frame, person_status_text, (x1, text_y_status), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # --- Ekipman Etiketlerini Belirle ---
                missing_items = []
                equipped_items = []
                if not has_helmet: missing_items.append("BARET")
                else: equipped_items.append("BARET")
                if not has_vest: missing_items.append("YELEK")
                else: equipped_items.append("YELEK")

                label_font_scale = 0.5
                label_thickness = 1
                label_padding = 5

                # --- Eksik Ekipman Etiketlerini Çiz (Kırmızı, Sağ Taraf) ---
                if not is_safe:
                    label_y_offset_right = text_height + 10 # Ana metnin biraz altına
                    label_margin_x_right = 10 # Kutunun sağına olan boşluk

                    for item_text in missing_items:
                        (label_w, label_h), label_bl = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
                        rect_x1 = x2 + label_margin_x_right
                        rect_y1 = y1 + label_y_offset_right
                        rect_x2 = rect_x1 + label_w + 2 * label_padding
                        rect_y2 = rect_y1 + label_h + 2 * label_padding
                        # Sınır kontrolü (frame dışına taşmasın)
                        if rect_x2 < w_frame and rect_y2 < h_frame:
                            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1) # Kırmızı
                            text_x = rect_x1 + label_padding
                            text_y_label = rect_y1 + label_h + label_padding
                            cv2.putText(frame, item_text, (text_x, text_y_label), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), label_thickness)
                        label_y_offset_right += label_h + 2 * label_padding + 5

                # --- Takılı Ekipman Etiketlerini Çiz (Yeşil, Sol Taraf) ---
                label_y_offset_left = text_height + 10 # Ana metnin biraz altına
                label_margin_x_left = 10 # Kutunun soluna olan boşluk

                for item_text in equipped_items:
                    (label_w, label_h), label_bl = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
                    # Etiket pozisyonu (kutunun soluna)
                    rect_x2 = x1 - label_margin_x_left
                    rect_y1 = y1 + label_y_offset_left
                    rect_x1 = rect_x2 - label_w - 2 * label_padding
                    rect_y2 = rect_y1 + label_h + 2 * label_padding

                    # Sınır kontrolü (frame dışına taşmasın)
                    if rect_x1 > 0 and rect_y2 < h_frame:
                        # Yeşil arka plan
                        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1) # Yeşil
                        # Siyah yazı
                        text_x = rect_x1 + label_padding
                        text_y_label = rect_y1 + label_h + label_padding
                        cv2.putText(frame, item_text, (text_x, text_y_label), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), label_thickness)
                    # Bir sonraki etiket için y offset'i artır
                    label_y_offset_left += label_h + 2 * label_padding + 5

            # --- Takipçi Temizliği ---
            # Bu frame'de görülmeyen ama hala takipte olanları kontrol et
            stale_ids = set(self.unsafe_persons_tracker.keys()) - processed_person_ids
            for stale_id in stale_ids:
                 # Ne kadar süredir görülmedi?
                 tracker_entry = self.unsafe_persons_tracker[stale_id]
                 time_since_last_seen = current_time - tracker_entry['timestamp'] # İlk görüldüğü zamandan beri geçen süre
                 last_report_time = tracker_entry.get('last_report_time', 0)
                 is_reported = tracker_entry.get('reported', False)
                 cooldown_active = is_reported and (current_time - last_report_time <= report_cooldown)

                 # Silme koşulları
                 # 1. Raporlanmadıysa ve uzun süre (delay*2) görülmediyse sil
                 if not is_reported and time_since_last_seen > self.report_delay * 2:
                     logger.debug(f"Takipteki {stale_id} uzun süredir görülmedi (raporlanmadı), takipten çıkarılıyor.")
                     del self.unsafe_persons_tracker[stale_id]
                 # 2. Raporlandıysa, soğuma süresi bittiyse VE uzun süre (delay*5) görülmediyse sil
                 elif is_reported and not cooldown_active and time_since_last_seen > self.report_delay * 5:
                     logger.debug(f"Raporlanan {stale_id} uzun süredir görülmedi (soğuma bitti), takipten çıkarılıyor.")
                     del self.unsafe_persons_tracker[stale_id]
                 # 3. Raporlandıysa ve soğuma süresi aktifken kişi kaybolduysa? Şimdilik tutuyoruz.
                 # else:
                 #    logger.debug(f"Stale ID {stale_id} durumu: reported={is_reported}, cooldown_active={cooldown_active}, time_since_last_seen={time_since_last_seen:.1f}s. Henüz silinmiyor.")


        except Exception as e:
             logger.error(f"Sonuç işleme hatası (process_detection_results): {e}", exc_info=True)
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
            logger.warning("Video akışı istendi ancak kamera aktif/açık değil.")
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
        sent_camera_off_frame = False # Flag to send the off frame only once

        while self.camera_active: # Döngü kontrolü
            # 1. Kameradan Frame Oku
            ret, frame = cap.read()
            if not ret:
                logger.warning("Kameradan frame okunamadı. Sonraki frame deneniyor...")
                time.sleep(0.1) # Kısa bir süre bekle
                continue # Döngünün başına dön, sonraki frame'i dene

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
            except Exception as e:
                 logger.error(f"Video akışı ana döngü hatası: {e}", exc_info=True)
                 # Hata durumunda döngüyü kırmak iyi olabilir
                 break # Döngüyü sonlandır

        # Döngü bitti (self.camera_active False oldu veya hata oluştu)
        logger.info("Video akış döngüsü sona erdi veya kamera kapatıldı.")

        # Son bir "Camera Off" frame'i gönder (eğer daha önce gönderilmediyse)
        if not sent_camera_off_frame:
            try:
                logger.info("Kamera kapatıldığı/döngü bittiği için son 'Camera Off' frame gönderiliyor.")
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
    # stay_safe_app = StaySafeApp(model_path=MODEL_PATH, db_path=DB_PATH) # db_path kaldırıldı
    stay_safe_app = StaySafeApp(model_path=MODEL_PATH)
    logger.info("StaySafeApp uygulaması başarıyla başlatıldı.")
    # Başlangıçta kamerayı otomatik açmak için:
    # try:
    #    stay_safe_app.toggle_camera()
    # except Exception as auto_start_err:
    #    logger.error(f"Başlangıçta kamera otomatik açılamadı: {auto_start_err}")

except FileNotFoundError as e:
    logger.error(f"Uygulama başlatılamadı - Gerekli dosya bulunamadı: {e}", exc_info=True)
    # stay_safe_app None kalacak, view'lar bunu kontrol etmeli
except Exception as e:
    logger.error(f"Uygulama başlatılırken kritik hata: {e}", exc_info=True)
    # stay_safe_app None kalacak

# --- Django Views ---
def check_app_status(func):
    """View fonksiyonları için stay_safe_app'in durumunu kontrol eden decorator."""
    def wrapper(request, *args, **kwargs):
        if stay_safe_app is None:
            logger.error(f"{func.__name__} çağrıldı ancak uygulama başlatılamamış.")
            # Hata durumuna uygun bir yanıt döndür
            if request.accepts('application/json'):
                return JsonResponse({'status': 'error', 'error': 'Uygulama düzgün başlatılamadı.'}, status=503) # Service Unavailable
            else:
                # Belki bir hata şablonu gösterilebilir
                 return render(request, 'display/error.html', {'error_message': 'Uygulama şu anda kullanılamıyor.'})
        return func(request, *args, **kwargs)
    return wrapper

def index(request):
     # Uygulama durumunu kontrol edelim (decorator kullanmadan)
     app_ready = stay_safe_app is not None
     camera_status = stay_safe_app.camera_active if app_ready else False
     context = {
         'camera_is_active': camera_status,
         'app_ready': app_ready,
         'error_message': None if app_ready else "Uygulama başlatılamadı. Lütfen kayıtları kontrol edin."
     }
     return render(request, 'display/index.html', context)

@check_app_status
def home(request):
    """Dashboard sayfasını gösterir."""
    app_ready = stay_safe_app is not None
    context = {
        'app_ready': app_ready,
        'error_message': None if app_ready else "Uygulama başlatılamadı. Lütfen kayıtları kontrol edin."
    }

    if not app_ready:
        return render(request, 'display/home.html', context)

    # Dashboard verilerini hazırla
    today = timezone.now().date()
    seven_days_ago = today - timedelta(days=6) # Son 7 günü kapsa

    # Özet Kartlar için Veriler
    total_reports = EmployeeReport.objects.count()
    today_reports = EmployeeReport.objects.filter(timestamp__date=today).count()
    # Buraya başka özet istatistikler eklenebilir (örn: en çok raporlanan çalışan vs.)

    # Grafik için Veriler (Son 7 gün)
    reports_per_day = EmployeeReport.objects.filter(timestamp__date__gte=seven_days_ago)\
                                         .annotate(day=TruncDate('timestamp'))\
                                         .values('day')\
                                         .annotate(count=Count('id'))\
                                         .order_by('day')

    # Grafik için etiketleri ve verileri hazırla
    # Son 7 günün tamamını göstermek için (rapor olmayan günler dahil)
    chart_labels = []
    chart_data = []
    report_counts_dict = {r['day']: r['count'] for r in reports_per_day}

    for i in range(7):
        current_day = seven_days_ago + timedelta(days=i)
        chart_labels.append(current_day.strftime('%d %b')) # Örn: 25 May
        chart_data.append(report_counts_dict.get(current_day, 0))

    # --- Yeni: Çalışan Bazlı Grafik Verileri (Top 5) ---
    employee_reports_top5 = EmployeeReport.objects.filter(employee__isnull=False)\
                                            .values('employee__name', 'employee__surname')\
                                            .annotate(
                                                full_name=Concat(F('employee__name'), Value(' '), F('employee__surname')),
                                                report_count=Count('id')
                                            )\
                                            .order_by('-report_count')[:5]

    employee_chart_labels = [item['full_name'] for item in employee_reports_top5]
    employee_chart_data = [item['report_count'] for item in employee_reports_top5]
    # --------------------------------------------------

    # Son Raporlar Listesi
    recent_reports = EmployeeReport.objects.all()[:5]

    context.update({
        'total_reports': total_reports,
        'today_reports': today_reports,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
        'recent_reports': recent_reports,
        # Yeni eklenen context değişkenleri
        'employee_chart_labels': json.dumps(employee_chart_labels),
        'employee_chart_data': json.dumps(employee_chart_data),
    })

    return render(request, 'display/home.html', context)

@check_app_status
def video_feed(request):
    """Video akışını sağlar."""
    if not stay_safe_app.camera_active:
        logger.info("Video akışı istendi ancak kamera kapalı.")
        # Kamera kapalıyken boş bir akış veya statik bir görüntü döndür
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame") # Boş akış döndürür

    # recognition=True ile nesne tespiti ve yüz tanıma akışını başlat
    return StreamingHttpResponse(stay_safe_app.get_video_stream(recognition=True),
                                 content_type="multipart/x-mixed-replace; boundary=frame")

@csrf_exempt
@check_app_status
def toggle_camera(request):
     """Kamera durumunu değiştirir."""
     if request.method == 'POST': # Sadece POST isteklerini kabul et
        try:
            is_active = stay_safe_app.toggle_camera()
            return JsonResponse({
                'status': 'success',
                'camera_is_active': is_active
            })
        except Exception as e:
            logger.error(f"Kamera durumu değiştirilirken view hatası: {e}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'error': f"Kamera durumu değiştirilemedi: {str(e)}"
            }, status=500)
     else:
        return JsonResponse({'status': 'error', 'error': 'Invalid request method'}, status=405)


@csrf_exempt
@check_app_status
def get_worker_info(request):
    """Son tanınan çalışanın bilgisini döndürür."""
    if request.method == 'GET': # Sadece GET isteklerini kabul et
        if not stay_safe_app.camera_active:
            return JsonResponse({'status': 'info', 'message': 'Kamera kapalı.'})

        # En son tanınan geçerli ismi al
        last_valid_name = None
        # predicted_names listesini tersten tarayarak ilk geçerli ismi bul
        for name in reversed(stay_safe_app.predicted_names):
            if name and name not in ["Unknown", "Error"]:
                last_valid_name = name
                break

        if last_valid_name:
            worker_info = stay_safe_app.findWorker(last_valid_name)
            return JsonResponse({'status': 'success', 'worker_info': worker_info})
        else:
             # Eğer listede hiç geçerli isim yoksa veya liste boşsa
             last_prediction = stay_safe_app.predicted_names[-1] if stay_safe_app.predicted_names else "None"
             return JsonResponse({'status': 'info', 'message': f'Geçerli çalışan tanınmadı. Son tahmin: {last_prediction}'})
    else:
         return JsonResponse({'status': 'error', 'error': 'Invalid request method'}, status=405)

def report_list(request):
    """
    Veritabanındaki tüm güvenlik raporlarını listeler.
    """
    reports = EmployeeReport.objects.all() # Tüm raporları al (ordering model meta'da tanımlı)
    context = {
        'reports': reports
    }
    return render(request, 'reports/report_list.html', context)

# İleride rapor detaylarını görmek için bir view daha eklenebilir:
# def report_detail(request, report_id):
#     report = get_object_or_404(EmployeeReport, pk=report_id)
#     context = {'report': report}
#     return render(request, 'reports/report_detail.html', context)