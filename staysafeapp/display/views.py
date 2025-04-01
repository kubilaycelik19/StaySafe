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
import sqlite3
import warnings
# from imutils.video import FPS # FPS kullanılmıyor gibi, kaldırılabilir
from ultralytics import YOLO
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile # Rapor görüntüsü kaydetmek için
from django.core.files.storage import default_storage # Dosya işlemleri için
from reports.models import EmployeeReport # Rapor modelini import et


# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
logger.info(f"Proje Ana Dizini (BASE_DIR): {BASE_DIR}")
logger.info(f"Statik Dosya Dizini (STATIC_DIR): {STATIC_DIR}")


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
DB_PATH = os.path.join(STATIC_DIR, "Workers.db")
NAMES_FILE = os.path.join(STATIC_DIR, 'names.json')
TRAINER_FILE = os.path.join(STATIC_DIR, 'trainer.yml')
# Haarcascade dosyasını OpenCV'nin kurulu olduğu yerden alındı
try:
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(CASCADE_PATH):
        logger.error(f"Haarcascade dosyası bulunamadı: {CASCADE_PATH}")
        # Alternatif bir yol veya hata verme stratejisi eklenebilir
        CASCADE_PATH = os.path.join(STATIC_DIR, 'haarcascade_frontalface_default.xml') # Statik klasörde de arandı
        logger.warning(f"Statik klasördeki cascade kullanılacak: {CASCADE_PATH}")

except AttributeError:
    logger.warning("cv2.data.haarcascades bulunamadı. Cascade yolu manuel olarak ayarlanmalı veya static klasörüne konulmalı.")
    CASCADE_PATH = os.path.join(STATIC_DIR, 'haarcascade_frontalface_default.xml') # Statik klasördeki cascade kullanılacak

REPORT_DELAY = 5 # Raporlama öncesi bekleme süresi (saniye)

# --- Veritabanı Sınıfı --- Düzenlenecek!!!
class WorkersDatabase:
    def __init__(self, db_name, default_table="employees"):
        self.db_name = db_name
        self.default_table = default_table
        # Veritabanı ve tablo yoksa oluştur
        if not os.path.exists(self.db_name):
            logger.warning(f"Veritabanı bulunamadı: {self.db_name}. Yeni veritabanı oluşturuluyor.")
            self.create_database(self.default_table)
            self.create_seed_data(self.default_table)
        else:
            # Veritabanı varsa tabloyu kontrol et
             conn = sqlite3.connect(self.db_name)
             cursor = conn.cursor()
             try:
                 cursor.execute(f"SELECT 1 FROM {self.default_table} LIMIT 1")
             except sqlite3.OperationalError:
                 logger.warning(f"'{self.default_table}' tablosu bulunamadı. Oluşturuluyor.")
                 self.create_database(self.default_table)
                 self.create_seed_data(self.default_table)
             finally:
                conn.close()


    def create_database(self, table_name):
        """Belirtilen tabloyu oluşturur."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                surname TEXT NOT NULL,
                age INTEGER NOT NULL
            );
            """)
            conn.commit()
            conn.close()
            logger.info(f"Veritabanı '{self.db_name}' ve tablo '{table_name}' başarıyla oluşturuldu/kontrol edildi.")
        except sqlite3.Error as e:
            logger.error(f"Veritabanı/tablo oluşturma hatası ({table_name}): {e}")

    def create_seed_data(self, table_name):
        """Eğer tablo boşsa, örnek çalışan verilerini ekler."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            if count == 0:
                employees = [
                    ("Emre", "Ozkan", 23),
                    ("Kubilay", "Celik", 25),
                    ("Zeynep", "Yilmaz", 35)
                ]
                cursor.executemany(f"INSERT INTO {table_name} (name, surname, age) VALUES (?, ?, ?)", employees)
                conn.commit()
                logger.info(f"Örnek veriler '{table_name}' tablosuna eklendi.")
            else:
                logger.info(f"'{table_name}' tablosu zaten veri içeriyor, örnek veri eklenmedi.")
            conn.close()
        except sqlite3.Error as e:
             logger.error(f"Örnek veri ekleme hatası ({table_name}): {e}")


    def find_employee(self, name):
        """Verilen isme göre çalışanı bulur."""
        employee = None
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            # İsme göre tam eşleşme arayalım (büyük/küçük harf duyarsız olabilir, COLLATE NOCASE eklenebilir)
            cursor.execute(f"SELECT * FROM {self.default_table} WHERE name=? COLLATE NOCASE", (name,))
            employee = cursor.fetchone()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Veritabanı hatası (find_employee '{name}'): {e}")
        return employee

# --- Yüz Tanıma Sınıfı ---
class FaceRecognitionSystem:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(CASCADE_PATH):
             logger.error(f"Yüz tespiti için cascade dosyası yüklenemedi: {CASCADE_PATH}")
             self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        self.names = {}
        self.cam = None
        self.model_loaded = False # Modelin laod kontrolü
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Yüz tanıma modeli ({TRAINER_FILE}) yüklenirken hata oluştu: {e}")
        self.load_names()
        # Kamera başlangıçta başlatılmıyor, toggle_camera ile yönetilecek

    def load_model(self):
        """Eğitilmiş yüz tanıma modelini yükler."""
        self.model_loaded = False
        if not os.path.exists(TRAINER_FILE):
            logger.warning(f"Trainer dosyası bulunamadı: {TRAINER_FILE}. Yüz tanıma yapılamayacak.")
            return

        try:
            self.recognizer.read(TRAINER_FILE)
            self.model_loaded = True
            logger.info("Yüz tanıma modeli (trainer.yml) başarıyla yüklendi.")
        except cv2.error as e:
             logger.error(f"Trainer dosyası ({TRAINER_FILE}) okunurken OpenCV hatası: {e}. Dosya bozuk veya uyumsuz olabilir.")
        except Exception as e:
            logger.error(f"Trainer dosyası ({TRAINER_FILE}) yüklenirken beklenmedik hata: {e}")

    def load_names(self):
        """ID-İsim eşleşmelerini JSON dosyasından yükler."""
        if not os.path.exists(NAMES_FILE):
            logger.warning(f"Names dosyası bulunamadı: {NAMES_FILE}. İsimler 'Unknown' olarak gösterilecek.")
            self.names = {}
            return

        try:
            with open(NAMES_FILE, 'r', encoding='utf-8') as fs: # utf-8 ekleyelim
                content = fs.read().strip()
                if content:
                    self.names = json.loads(content)
                    logger.info(f"İsim eşleşmeleri ({NAMES_FILE}) yüklendi: {self.names}")
                else:
                    logger.warning(f"Names dosyası ({NAMES_FILE}) boş.")
                    self.names = {}
        except json.JSONDecodeError as e:
             logger.error(f"Names dosyası ({NAMES_FILE}) JSON formatında değil: {e}")
             self.names = {}
        except Exception as e:
            logger.error(f"Names dosyası ({NAMES_FILE}) yüklenirken hata: {e}")
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
            # Kamera ayarlarının gerçekten uygulanıp uygulanmadığını kontrol etme işlemi. Extra kameralar destekliyor mu kontrol edilecek.
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
        """Verilen görüntüdeki yüzü tanır."""
        name = "Unknown"
        confidence_score = 0

        if not self.model_loaded:
             #logger.debug("Yüz tanıma modeli yüklenmediği için tanıma yapılamıyor.")
             return name, confidence_score # Model yoksa direkt Unknown dön

        if self.face_cascade is None:
            logger.warning("Face cascade yüklenmediği için yüz tespiti yapılamıyor.")
            return name, confidence_score

        try:
            # Görüntü zaten BGR ise tekrar dönüştürmeye gerek yok, ancak ROI gri olmalı
            if len(img.shape) == 3 and img.shape[2] == 3:
                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                 gray = img # Zaten gri ise kullan
            else:
                 logger.warning(f"Beklenmedik görüntü formatı: shape={img.shape}")
                 return "Error", 0


            # Yüz tespiti (ROI üzerinde değil, gelen orijinal ROI üzerinde)
            faces = self.face_cascade.detectMultiScale(
                gray, # Gri tonlamalı görüntü kullanılmalı
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            if len(faces) > 0:
                # En büyük yüzü secme islemi (genellikle en belirgin olanı)
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]

                # Yüz ROI'sini al (gri görüntüden)
                face_roi_gray = gray[y:y+h, x:x+w]

                if face_roi_gray.size == 0:
                    logger.warning("Tespit edilen yüz ROI'si boş.")
                    return name, confidence_score

                try:
                    # Tanıma yap
                    id_recognized, confidence = self.recognizer.predict(face_roi_gray)

                    # Confidence: Düşük değer daha iyi eşleşme (0 mükemmel eşleşme)
                    # 0-100 arası bir güven skoruna çevir
                    # Eşik değeri (örn. 70-80) threshold 
                    if confidence < 80: # Eşik değeri - daha düşükse daha güvenli
                        name = self.names.get(str(id_recognized), "Unknown")
                        confidence_score = round((1 - (confidence / 100)) * 100) # Basit bir dönüşüm
                        # logger.debug(f"Tanınan ID: {id_recognized}, İsim: {name}, Confidence(LBPH): {confidence:.2f}, Skor: {confidence_score}")
                    else:
                         name = "Unknown"
                         confidence_score = round((1 - (confidence / 100)) * 100) # Düşük skor
                         # logger.debug(f"Tanınan ID: {id_recognized}, Confidence çok yüksek: {confidence:.2f}. 'Unknown' kabul edildi.")


                except cv2.error as cv_err:
                    # Tanıma sırasında hata (örn. model uyumsuzluğu)
                    logger.warning(f"OpenCV predict hatası: {cv_err}. Model veya ROI ile ilgili sorun olabilir.")
                    name = "Error"
                    confidence_score = 0
                except Exception as pred_err:
                    logger.error(f"Yüz tanıma (predict) sırasında beklenmedik hata: {pred_err}")
                    name = "Error"
                    confidence_score = 0
            # else: # Yüz tespit edilemediyse
                # logger.debug("Görüntüde yüz tespit edilemedi.")


        except Exception as e:
            logger.error(f"Yüz tanıma genel hata: {e}")
            name = "Error"
            confidence_score = 0

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
    def __init__(self, model_path: str, db_path: str, width=640, height=480):
        self.model_path = model_path
        self.db_path = db_path
        self.width = width # YOLO'nun işleyeceği genişlik
        self.height = height # Kamera yüksekliği
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Kullanılan cihaz: {self.device}")

        if not os.path.exists(self.model_path):
            logger.error(f"YOLO modeli bulunamadı: {self.model_path}")
            raise FileNotFoundError(f"Gerekli YOLO modeli bulunamadı: {self.model_path}")
        self.model = self.create_yolo_model()

        self.database = WorkersDatabase(db_name=self.db_path)
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
        """Verilen isimdeki çalışanı veritabanında arar ve cache kullanır."""
        if not name or name in ["Unknown", "Error"]:
            return "Tanımsız veya hatalı çalışan adı."

        # Cache kontrolü
        if name in self.worker_info_cache:
            return self.worker_info_cache[name]

        try:
            worker = self.database.find_employee(name=name)
            if worker:
                # worker bir tuple (id, name, surname, age)
                info = f"Çalışan: {worker[1]} {worker[2]} (ID: {worker[0]}, Yaş: {worker[3]})"
                self.worker_info_cache[name] = info # Cache'e ekle
                return info
            else:
                 info = f"'{name}' isimli çalışan bulunamadı."
                 self.worker_info_cache[name] = info # Bulunamadı bilgisini de cache'le
                 return info
        except Exception as e:
            logger.error(f"Çalışan arama hatası ({name}): {e}")
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

            # Raporu oluştur
            report = EmployeeReport(
                employee=employee_instance,
                is_equipped=False, # Rapor ekipmansızlık durumu için
                image=image_content,
                location="Kamera Görüntüsü", # Varsayılan konum
                missing_equipment=missing_equipment_str # Eksik ekipmanları kaydet
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
                             person_status_text += f" (Raporlandı - {remaining_cooldown}s)"
                        elif tracker_entry['reported']:
                             person_status_text += " (Raporlandı)" # Soğuma bitti ama hala güvensizse
                        elif time_elapsed > 1:
                            person_status_text += f" ({int(time_elapsed)}s)" # Kaç sn güvensiz

                # --- Çizim --- 
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                (text_width, text_height), baseline = cv2.getTextSize(person_status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 5
                # Arka plan ekleyelim
                cv2.rectangle(frame, (x1, text_y - text_height - baseline), (x1 + text_width, text_y + baseline), (0,0,0), -1)
                cv2.putText(frame, person_status_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

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

                 # Silme koşullarını gözden geçirelim:
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
        try:
            if self.camera_active:
                logger.info("Kamera kapatılıyor...")
                self.frame_processor.stop()
                self.face_recognizer.release_camera()
                self.camera_active = False
                self.predicted_names = []
                self.worker_info_cache = {}
                self.unsafe_persons_tracker.clear() # Kapatırken takip listesini temizle
                logger.info("Kamera ve frame işleyici başarıyla durduruldu.")
                return False
            else:
                # Kamerayı başlat
                logger.info("Kamera başlatılıyor...")
                if not self.face_recognizer.initialize_camera():
                     logger.error("Kamera başlatılamadı.")
                     self.camera_active = False # Başlatılamazsa durumu false yap
                     return False # Başlatılamadığını belirt

                 # Kamera başarılıysa Frame işlemciyi başlat
                self.frame_processor.start()
                self.camera_active = True
                logger.info("Kamera ve frame işleyici başarıyla başlatıldı.")
                return True
        except Exception as e:
            logger.error(f"Kamera durumu değiştirilirken hata: {e}", exc_info=True)
            # Hata durumunda kaynakları serbest bırakmaya çalışalım
            try:
                self.frame_processor.stop()
                self.face_recognizer.release_camera()
            except Exception as cleanup_err:
                 logger.error(f"Hata sonrası temizlik sırasında ek hata: {cleanup_err}")
            self.camera_active = False # Hata durumunda kapalı olarak işaretle
            self.unsafe_persons_tracker.clear() # Hata durumunda da temizle
            raise Exception(f"Kamera durumu değiştirilemedi: {str(e)}") # Hatayı yukarıya ilet


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

        logger.info("Video akışı başlıyor...")
        cap = self.face_recognizer.cam

        while self.camera_active: # Döngü kontrolü
            # 1. Kameradan Frame Oku
            ret, frame = cap.read()
            if not ret:
                logger.warning("Kameradan frame okunamadı. Akış durduruluyor.")
                # Belki yeniden başlatmayı denemek yerine akışı durdurmak daha iyi
                self.toggle_camera() # Kamerayı kapatmayı dene
                break # Döngüden çık

            frame = cv2.flip(frame, 1) # Görüntüyü aynala

            # 2. Frame'i İşleme Kuyruğuna Gönder
            try:
                self.frame_queue.put_nowait(frame) # Bloklamayan put
            except queue.Full:
                # Kuyruk doluysa en eskiyi atıp yeniyi ekleyelim (canlılık önemliyse)
                 try:
                    self.frame_queue.get_nowait() # Eski frame'i at
                    self.frame_queue.put_nowait(frame) # Yeni frame'i ekle
                    # logger.debug("Frame kuyruğu doluydu, eski frame atıldı.")
                 except queue.Empty: pass # Boşsa sorun yok
                 except queue.Full: # Tekrar doluysa (çok hızlı frame geliyorsa)
                     logger.warning("Frame kuyruğu anlık olarak çok dolu, frame atlanıyor.")
                     pass # Bu frame'i atla

            # 3. İşlenmiş Sonucu Al
            try:
                # Timeout ile al, result yoksa takılma
                processed_frame, results = self.result_queue.get(timeout=0.5)

                # 4. Sonuçları İşle ve Görüntüyü Hazırla
                if recognition and processed_frame is not None:
                     # Nesne tespiti ve yüz tanımayı içeren işlem
                     output_frame = self.process_detection_results(processed_frame, results)
                elif processed_frame is not None:
                     # Sadece frame'i göster (recognition=False ise veya result yoksa)
                     output_frame = processed_frame
                else:
                     # Hata veya boş sonuç durumunda orijinal frame'i kullan (veya hata mesajı ekle)
                     output_frame = frame # Hata durumunda orijinal frame'i gönderelim
                     cv2.putText(output_frame, "Processing Issue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


                # 5. Frame'i JPEG'e Çevir ve Gönder
                ret_encode, buffer = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) # Kaliteyi düşürelim
                if ret_encode:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    logger.warning("Frame JPEG formatına dönüştürülemedi.")

            except queue.Empty:
                # Sonuç kuyruğu boşsa, bir süre bekle ve devam et
                # logger.debug("Sonuç kuyruğu boş, bekleniyor...")
                time.sleep(0.02) # CPU'yu yormamak için küçük bir bekleme
                continue
            except Exception as e:
                 logger.error(f"Video akışı ana döngü hatası: {e}", exc_info=True)
                 # Hata durumunda döngüyü kırmak iyi olabilir
                 break # Döngüyü sonlandır

        logger.info("Video akış döngüsü sona erdi.")
        # Kamera kapalıysa veya hata oluştuysa toggle_camera zaten kaynakları temizlemiş olmalı


# --- Global StaySafeApp Nesnesi ---
# Bu nesne Django uygulaması başlatıldığında bir kere oluşturulur.
stay_safe_app = None
try:
    logger.info("StaySafeApp uygulaması başlatılıyor...")
    stay_safe_app = StaySafeApp(model_path=MODEL_PATH, db_path=DB_PATH)
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

# @check_app_status # home için gerekli değil gibi
def home(request):
    return render(request, 'display/home.html')

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