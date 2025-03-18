import cv2
import os
import dlib
import time
import numpy as np
from datetime import datetime

def check_environment():
    """Environment kontrolü"""
    print("\nKütüphane sürümleri:")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"dlib: {dlib.__version__}")
    print("-" * 30)

# Environment kontrolü
check_environment()

# Numpy sürüm kontrolü
np_version = np.__version__
if not np_version.startswith('1.23'):
    print(f"Uyarı: Numpy sürümünüz ({np_version}) önerilen sürümden (1.23.x) farklı.")
    print("Bu durum görüntü işleme sırasında sorunlara neden olabilir.")
    response = input("Devam etmek istiyor musunuz? (e/h): ")
    if response.lower() != 'e':
        print("Program sonlandırılıyor...")
        exit()

class FaceDataCollector:
    def __init__(self, save_dir="./dataset/faces", capture_interval=0.5, max_images=100):
        """
        Yüz veri toplayıcı sınıfı
        
        Args:
            save_dir (str): Görüntülerin kaydedileceği ana dizin
            capture_interval (float): İki görüntü alma arasındaki minimum süre (saniye)
            max_images (int): Toplanacak maksimum görüntü sayısı
        """
        self.save_dir = save_dir
        self.capture_interval = capture_interval
        self.max_images = max_images
        self.detector = dlib.get_frontal_face_detector()
        
    def setup_camera(self):
        """Kamera bağlantısını kur ve kontrol et"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Kamera açılamadı! Lütfen bağlantıyı kontrol edin.")
        return cap
    
    def create_person_directory(self, name):
        """Kişi için klasör oluştur"""
        person_dir = os.path.join(self.save_dir, name.lower())
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"'{name}' için klasör oluşturuldu: {person_dir}")
        else:
            print(f"'{name}' için klasör zaten mevcut: {person_dir}")
        return person_dir
    
    def save_face_image(self, frame, face, save_path, count):
        """Tespit edilen yüzü kaydet"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Yüz koordinatlarının görüntü sınırları içinde olduğunu kontrol et
        if (0 <= y < frame.shape[0] and 0 <= y + h < frame.shape[0] and 
            0 <= x < frame.shape[1] and 0 <= x + w < frame.shape[1]):
            
            try:
                # Orijinal frame BGR formatında
                face_img = frame[y:y + h, x:x + w].copy()
                
                # Görüntüyü yeniden boyutlandır
                face_img = cv2.resize(face_img, (224, 224))
                
                # Timestamp ekle
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = os.path.join(save_path, f"{count}_{timestamp}.jpg")
                
                # Görüntüyü kaydet
                success = cv2.imwrite(image_name, face_img)
                
                if success:
                    print(f"Görüntü kaydedildi: {image_name}")
                    return True
                else:
                    print("Görüntü kaydedilemedi!")
                    return False
                    
            except Exception as e:
                print(f"Görüntü kaydedilirken hata oluştu: {str(e)}")
                return False
                
        return False
    
    def collect_face_data(self, person_name):
        """Kişi için yüz verisi topla"""
        try:
            # Kamera ve klasör hazırlığı
            cap = self.setup_camera()
            save_path = self.create_person_directory(person_name)
            
            count = 0
            last_capture_time = time.time()
            
            print("\nVeri toplama başlıyor...")
            print(f"Hedef: {self.max_images} görüntü")
            print("Çıkmak için 'q' tuşuna basın")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Görüntü alınamadı!")
                

                frame = cv2.flip(frame, 1)
                
                # Yüz tespiti için RGB'ye dönüştür
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector(rgb_frame)
                
                for face in faces:
                    # Yüzün etrafına kutu çiz
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Yeterli süre geçtiyse yeni görüntü kaydet
                    if time.time() - last_capture_time >= self.capture_interval:
                        if self.save_face_image(frame, face, save_path, count):
                            count += 1
                            last_capture_time = time.time()
                
                # Bilgileri ekranda göster
                info_text = f"Toplanan: {count}/{self.max_images}"
                cv2.putText(frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Veri Toplama", frame)
                
                # Çıkış kontrolü
                if cv2.waitKey(1) & 0xFF == ord('q') or count >= self.max_images:
                    break
            
            print(f"\nToplam {count} görüntü kaydedildi.")
            
        except Exception as e:
            print(f"Hata: {str(e)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    # Veri toplama işlemi
    if not os.path.exists("./dataset/faces"):
        os.makedirs("./dataset/faces")
    
    collector = FaceDataCollector(
        save_dir="./dataset/faces",
        capture_interval=0.5,
        max_images=100
    )
    
    while True:
        person_name = input("\nKişi adını girin (Çıkmak için 'q'): ").strip()
        
        if person_name.lower() == 'q':
            break
        
        if person_name:
            collector.collect_face_data(person_name)
        else:
            print("Geçerli bir isim girmediniz!")
    
    print("\nProgram sonlandırıldı.")

if __name__ == "__main__":
    main()