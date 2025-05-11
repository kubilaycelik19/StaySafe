import cv2

def test_camera(camera_index=0):
    print(f"Kamera indeksi {camera_index} test ediliyor...")
    cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW) # CAP_DSHOW ile deneyelim

    if not cap.isOpened():
        print(f"HATA: Kamera indeksi {camera_index} (CAP_DSHOW ile) AÇILAMADI.")
        print("Varsayılan backend deneniyor...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"HATA: Kamera indeksi {camera_index} (varsayılan ile de) AÇILAMADI.")
            return

    print(f"Kamera indeksi {camera_index} başarıyla AÇILDI.")
    
    ret, frame = cap.read()
    
    if ret:
        print("İlk frame başarıyla OKUNDU.")
        # cv2.imshow(f'Kamera Testi (Index {camera_index})', frame)
        # print("Bir pencere açıldı. Kapatmak için 5 saniye bekleyin veya pencereye tıklayıp bir tuşa basın.")
        # cv2.waitKey(5000) 
        print("GUI fonksiyonları (imshow, waitKey) test amaçlı devre dışı bırakıldı.")
    else:
        print(f"HATA: Kamera indeksi {camera_index} açıldı AMA frame OKUNAMADI.")

    cap.release()
    print(f"Kamera indeksi {camera_index} serbest bırakıldı.")
    # cv2.destroyAllWindows()
    print("Test tamamlandı.")

if __name__ == '__main__':
    # Farklı kamera indekslerini test edebilirsiniz.
    # Genellikle 0, 1 veya -1 (otomatik seçim) kullanılır.
    test_camera(0) 
    # test_camera(1)
