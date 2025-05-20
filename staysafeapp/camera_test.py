import cv2
import time

def test_camera():
    """Kamerayı test ediyorum."""
    # Kamerayı aç
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return
    
    print("Kamera başarıyla açıldı!")
    print("Test için 5 saniye bekleyin...")
    
    # 5 saniye boyunca görüntü al
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Kamera Testi', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    print("Test tamamlandı!")

if __name__ == "__main__":
    test_camera()
