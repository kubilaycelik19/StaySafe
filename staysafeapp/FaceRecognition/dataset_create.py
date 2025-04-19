# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import os
import numpy as np
import logging
import time
import argparse
import sys

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ayarlar ve Sabitler ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Veri seti dizini (Bu script'e göre ayarlandı)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.dirname(SCRIPT_DIR) # staysafeapp klasörü varsayımı
DATASET_DIR = os.path.join(BASE_PROJECT_DIR, 'static', 'dataset')

if not os.path.exists(DATASET_DIR):
    logger.info(f"Veri seti dizini oluşturuluyor: {DATASET_DIR}")
    os.makedirs(DATASET_DIR)

# MediaPipe Face Mesh ayarları
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
except AttributeError:
    logger.error("MediaPipe Face Mesh başlatılamadı. MediaPipe kurulu mu?")
    mp_face_mesh = None
    face_mesh = None
    sys.exit(1) # MediaPipe yoksa devam etme
except Exception as e:
    logger.error(f"MediaPipe başlatılırken hata: {e}")
    mp_face_mesh = None
    face_mesh = None
    sys.exit(1)

# --- Veri Seti Oluşturma Fonksiyonu ---
def create_dataset(person_name, max_frames=200, interval=0.5):
    """Belirtilen kişi için yüz görüntülerini toplar."""
    if not mp_face_mesh or not face_mesh:
        logger.error("MediaPipe başlatılamadığı için veri seti oluşturulamıyor.")
        return

    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        logger.info(f"Kişi dizini oluşturuldu: {person_dir}")
    else:
        # Mevcut dizin varsa kullanıcıya soralım
        overwrite = input(f"'{person_name}' için dizin zaten mevcut ({person_dir}).\nİçeriğin üzerine yazılsın mı? (e/H): ").lower()
        if overwrite != 'e':
            logger.info("İşlem iptal edildi.")
            return
        else:
             logger.warning(f"Mevcut dizinin üzerine yazılacak: {person_dir}")
             # Eski dosyaları silmek isteğe bağlı eklenebilir

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Kamera {CAMERA_INDEX} açılamadı.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_count = 0
    last_save_time = 0

    logger.info(f"'{person_name}' için kayıt başlıyor. Kameraya bakın. (Çıkmak için 'q')")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Kameradan frame alınamadı.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1) # Aynalama
        display_frame = frame.copy() # Üzerine çizim yapmak için kopya

        # Yüz tespiti
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        face_detected_in_frame = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            
            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
            
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            face_detected_in_frame = True

            # Kayıt
            current_time = time.time()
            if (current_time - last_save_time) >= interval:
                face_img = frame[y_min:y_max, x_min:x_max]
                if face_img.size > 0:
                    img_path = os.path.join(person_dir, f'face_{frame_count}.jpg')
                    cv2.imwrite(img_path, face_img)
                    logger.info(f"Yüz kaydedildi: {img_path} ({frame_count + 1}/{max_frames})")
                    frame_count += 1
                    last_save_time = current_time

        # Durum bilgisini göster
        status_text = f"Kaydedilen: {frame_count}/{max_frames}"
        if not face_detected_in_frame:
             status_text += " (Yuz bulunamadi! Kameraya yaklasin)"
             cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
             cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f'{person_name} icin veri toplama (Cikis: q)', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Kullanıcı tarafından işlem durduruldu.")
            break

    logger.info(f"Veri toplama tamamlandı. {frame_count} görüntü kaydedildi: {person_dir}")
    cap.release()
    cv2.destroyAllWindows()

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Belirtilen kişi için yüz veri seti oluşturur.')
    parser.add_argument('name', type=str, help='Veri setine eklenecek kişinin adı (örn: "Emre Ozkan"). Boşluk içeriyorsa tırnak içine alın.')
    parser.add_argument('--frames', type=int, default=150, help='Kaydedilecek maksimum frame sayısı.')
    parser.add_argument('--interval', type=float, default=0.3, help='Frame kaydetme aralığı (saniye).')
    
    args = parser.parse_args()
    
    create_dataset(args.name, max_frames=args.frames, interval=args.interval)
