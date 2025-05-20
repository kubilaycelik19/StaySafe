import face_recognition  # Yüz tanıma işlemleri için kullanılan kütüphane
import os, sys           # Dosya ve sistem işlemleri için
import cv2               # OpenCV, görüntü işleme için
import numpy as np       # Sayısal işlemler için
import math              # Matematiksel işlemler için

def face_confidence(face_distance, face_match_threshold=0.6):
    """Yüz mesafesini güven skoruna çeviriyorum."""
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        # Eşik değerinden büyükse doğrusal olarak güven skoru döndür
        return str(round(linear_val * 100, 2)) + '%'
    else:
        # Eşik değerinden küçükse daha karmaşık bir formülle güven skoru döndür
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def preprocess_image(image):
    """Görüntüyü ön işlemden geçiriyorum."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram eşitleme ve gürültü giderme istenirse açılabilir
    """equalized = cv2.equalizeHist(gray)
    denoised = cv2.fastNlMeansDenoising(equalized)"""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

class FaceRecognition:
    """Yüz tanıma işlemlerini yöneten ana sınıf."""
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True  # Her karede işlem yapılacak mı?

    def __init__(self):
        """Sınıfı başlatıp yüz kodlamalarını oluşturuyorum."""
        self.encode_faces()

    def encode_faces(self):
        """Faces klasöründeki yüzleri kodluyorum."""
        for person_folder in os.listdir('faces'):
            person_path = os.path.join('faces', person_folder)
            
            if not os.path.isdir(person_path):
                continue  # Sadece klasörler işlenir
                
            person_encodings = []
            
            for image_file in os.listdir(person_path):
                # Sadece resim dosyalarını işle
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(person_path, image_file)
                face_image = face_recognition.load_image_file(image_path)  # Resmi yükle
                face_image = preprocess_image(face_image)  # Ön işlem uygula
                
                face_encodings = face_recognition.face_encodings(face_image)  # Yüz kodlamasını al
                
                if face_encodings:
                    person_encodings.append(face_encodings[0])  # İlk yüz kodlamasını ekle
            
            if person_encodings:
                # Kişinin tüm resimlerinden ortalama bir yüz kodlaması oluştur
                avg_encoding = np.mean(person_encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_folder)  # Kişinin adını ekle

        print('Yüz kodlamaları tamamlandı!')  # Kodlama tamamlandı mesajı
        print(f'Bilinen yüzler: {self.known_face_names}')  # Bilinen yüzler

    def run_recognition(self):
        """Kamera görüntüsü üzerinde yüz tanıma işlemini başlatıyorum."""
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Hata: Kamera açılamadı.")  # Kamera açılamazsa çık

        while True:
            ret, frame = video_capture.read()  # Bir kare oku
            frame = cv2.flip(frame, 1)  # Aynalama (selfie gibi)

            if self.process_current_frame:
                frame = preprocess_image(frame)  # Ön işlem uygula
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Küçült
                rgb_small_frame = small_frame[:, :, ::-1]  # BGR'den RGB'ye çevir

                # Küçük karede yüz konumlarını ve kodlamalarını bul
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # Bilinen yüzlerle karşılaştır
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = 'unknown'

                    # En yakın yüzü bul
                    # Arkaplanda => np.linalg.norm(face_encodings - face_to_compare, axis=1) işlemini yapıyor.
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding) 
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        # Eşleşme varsa isim ve güven skorunu al
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} {confidence}')  # Sonuçları ekle

            self.process_current_frame = not self.process_current_frame  # Her karede işlem yapma

            # Yüzlerin etrafına kutu çiz ve isimleri yaz
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Yüz kutusu
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)  # Alt kutu
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)  # İsim ve güven

            cv2.imshow('Yüz Tanıma', frame)  # Sonucu göster

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # 'q' tuşuna basılırsa çık

        video_capture.release()  # Kamerayı serbest bırak
        cv2.destroyAllWindows()  # Pencereleri kapat

if __name__ == "__main__":
    fr = FaceRecognition()  # Sınıfı başlat
    fr.run_recognition()    # Yüz tanıma döngüsünü başlat
