from ultralytics import YOLO
import torch
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Eğitim {torch.cuda.get_device_name(0)} üzerinde yapılacak')
        print(f'Kullanılabilir CUDA belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    model = YOLO('yolo11m.pt')

    results = model.train(
        data='C:/Users/celik/Desktop/StaySafe/YOLO/safetyDataset/data.yaml',
        epochs=50,
        imgsz=640,
        # batch=16,
        device='cuda',
        name='safety_model50epochs',
        # augment=True,
        # # Temel augmentasyonlar
        # hsv_h=0.015,  # renk tonu değişimi
        # hsv_s=0.3,    # doygunluk değişimi
        # hsv_v=0.2,    # parlaklık değişimi
        # degrees=0.3,  # döndürme
        # translate=0.1, # çevirme
        # scale=0.2,    # ölçekleme
        # shear=0.1,    # kaydırma
        # flipud=0.2,   # dikey çevirme
        # fliplr=0.5,   # yatay çevirme
        # mosaic=0.3,   # mozaik
        # mixup=0.1,    # mixup
        
    )

if __name__ == '__main__':
    main() 