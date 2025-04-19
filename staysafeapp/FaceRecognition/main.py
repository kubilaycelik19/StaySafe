# -*- coding: utf-8 -*-
import argparse
import sys

# --- Eğitim fonksiyonları için importlar --- 
# Bu fonksiyonları artık main() içinde, mod seçildikten sonra import edeceğiz.
# try:
#     from .train.arcface_train import train_arcface
#     from .train.facenet_train import train_facenet
# except ImportError as e:
#     print(f"HATA: Gerekli eğitim modülleri import edilemedi. Scriptlerin doğru dizinlerde olduğundan emin olun. Hata: {e}")
#     sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='StaySafe Yüz Tanıma Veri Seti ve Eğitim Yöneticisi')
    
    subparsers = parser.add_subparsers(dest='mode', help='Çalışma Modu', required=True)
    
    # --- Veri Seti Oluşturma Modu ---
    parser_dataset = subparsers.add_parser('dataset', help='Yeni bir kişi için yüz veri seti oluşturur.')
    parser_dataset.add_argument('name', type=str, help='Veri setine eklenecek kişinin adı (örn: "Emre Ozkan").')
    parser_dataset.add_argument('--frames', type=int, default=150, help='Kaydedilecek maksimum frame sayısı.')
    parser_dataset.add_argument('--interval', type=float, default=0.3, help='Frame kaydetme aralığı (saniye).')
    
    # --- ArcFace Eğitim Modu ---
    parser_train_arcface = subparsers.add_parser('train-arcface', help='Mevcut veri setini kullanarak ArcFace modelini eğitir.')
    parser_train_arcface.add_argument('--epochs', type=int, default=50, help='Eğitim için epoch sayısı.')
    parser_train_arcface.add_argument('--batch_size', type=int, default=32, help='Eğitim için batch boyutu.')
    parser_train_arcface.add_argument('--lr', type=float, default=0.001, help='Eğitim için öğrenme oranı.')

    # --- FaceNet Eğitim Modu ---
    parser_train_facenet = subparsers.add_parser('train-facenet', help='Mevcut veri setini kullanarak FaceNet modelini eğitir.')
    parser_train_facenet.add_argument('--epochs', type=int, default=50, help='Eğitim için epoch sayısı.')
    parser_train_facenet.add_argument('--batch_size', type=int, default=32, help='Eğitim için batch boyutu.')
    parser_train_facenet.add_argument('--lr', type=float, default=0.001, help='Eğitim için öğrenme oranı.')
    
    args = parser.parse_args()
    
    # Seçilen moda göre ilgili fonksiyonu çağır
    if args.mode == 'dataset':
        try:
            from .dataset_create import create_dataset
            print(f"Veri seti oluşturma başlatılıyor: Kişi='{args.name}', Frames={args.frames}, Interval={args.interval}")
            create_dataset(args.name, max_frames=args.frames, interval=args.interval)
        except ImportError as e:
            print(f"HATA: dataset_create modülü import edilemedi. Hata: {e}")
            sys.exit(1)
    
    elif args.mode == 'train-arcface':
        try:
            # Sadece bu mod seçildiğinde import et
            from .train.arcface_train import train_arcface 
            print(f"ArcFace eğitimi başlatılıyor: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
            train_arcface(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        except ImportError as e:
             print(f"HATA: ArcFace eğitim modülü import edilemedi. Hata: {e}")
             sys.exit(1)

    elif args.mode == 'train-facenet':
        try:
            # Sadece bu mod seçildiğinde import et
            from .train.facenet_train import train_facenet 
            print(f"FaceNet eğitimi başlatılıyor: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
            # Ortam kontrolü facenet_train.py içinde yapılıyor.
            train_facenet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        except ImportError as e:
            print(f"HATA: FaceNet eğitim modülü import edilemedi (facenet-pytorch kurulu mu?). Hata: {e}")
            sys.exit(1)
            
    else:
        parser.print_help()

# --- Ana Çalıştırma Bloğu ---
# Örnek Kullanımlar:
# python -m FaceRecognition.main dataset "Kişi Adı Soyadı" --frames 200 --interval 0.4
# python -m FaceRecognition.main train-arcface --epochs 100 --batch_size 16 --lr 0.0005
# python -m FaceRecognition.main train-facenet --epochs 100 --batch_size 16 --lr 0.0005

if __name__ == "__main__":
    main()
