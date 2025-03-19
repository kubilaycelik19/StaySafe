import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import winsound

def play_notification():
    frequency = 1000
    duration = 1000
    for _ in range(3):
        winsound.Beep(frequency, duration)

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FaceRecognitionTrainer:
    def __init__(self, data_dir, batch_size=16, num_epochs=50, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dengeli veri artırma
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation için basit dönüşümler
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Veri yükleyicileri
        self.train_dataset = FaceDataset(os.path.join(data_dir, 'train'), self.train_transform)
        self.val_dataset = FaceDataset(os.path.join(data_dir, 'val'), self.val_transform)
        self.test_dataset = FaceDataset(os.path.join(data_dir, 'test'), self.val_transform)
        
        print(f"Eğitim seti boyutu: {len(self.train_dataset)}")
        print(f"Validation seti boyutu: {len(self.val_dataset)}")
        print(f"Test seti boyutu: {len(self.test_dataset)}")
        
        # Veri yükleme optimizasyonları
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,  # Worker sayısını artır
            pin_memory=True,
            persistent_workers=True,  # Worker'ları sürekli aktif tut
            prefetch_factor=2  # Önceden yükleme
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Model ve loss fonksiyonu
        self.num_classes = len(self.train_dataset.classes)
        self.model = self.create_model()
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer ve scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Eğitim geçmişi
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def create_model(self):
        # Mobilenet_v3_small modelini kullan (daha hafif)
        model = models.mobilenet_v3_small(pretrained=True)
        
        # Son katmanı güncelle
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, self.num_classes)
        )
        
        return model.to(self.device)
    
    def train(self):
        print("\nEğitim başlıyor...")
        print("Model yapılandırması:")
        print("- Model: MobileNet_v3_small (pretrained)")
        print("- Giriş boyutu: 224x224")
        print(f"- Sınıf sayısı: {self.num_classes}")
        print("- Dropout: 0.3")
        print("- Weight decay: 0.01")
        print("- Label smoothing: 0.2")
        
        best_val_acc = 0.0
        # patience = 10
        # patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Eğitim
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': f'{train_loss/len(self.train_loader):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Doğrulama
            val_loss, val_acc = self.validate()
            
            # Learning rate güncelleme
            self.scheduler.step(val_acc)
            
            # Early stopping ve model kaydetme
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # patience_counter = 0
                self.save_model()
                print(f'\nYeni en iyi model! (Val Acc: {val_acc:.2f}%)')
            else:
                # patience_counter += 1
                # print(f'\nİyileşme yok. Patience: {patience_counter}/{patience}')
                # if patience_counter >= patience:
                #     print(f'\nEarly stopping! En iyi validation accuracy: {best_val_acc:.2f}%')
                #     break
                print(f'\nİyileşme yok. En iyi accuracy: {best_val_acc:.2f}%')
            
            # Metrikleri kaydet
            self.train_losses.append(train_loss / len(self.train_loader))
            self.val_losses.append(val_loss)
            self.train_accs.append(100. * train_correct / train_total)
            self.val_accs.append(val_acc)
        
        self.plot_training_history()
        play_notification()
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * val_correct / val_total
        return val_loss, val_acc
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': {idx: cls for idx, cls in enumerate(self.train_dataset.classes)},
            'val_acc': self.val_accs[-1] if self.val_accs else 0.0
        }, 'best_face_model.pth')
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def test(self):
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_bar = tqdm(self.test_loader, desc='Test')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                test_bar.set_postfix({
                    'loss': f'{test_loss/len(self.test_loader):.4f}',
                    'acc': f'{100.*test_correct/test_total:.2f}%'
                })
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * test_correct / test_total
        print(f'\nTest sonuçları:')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

if __name__ == "__main__":
    # Veri seti dizini
    data_dir = "C:/Users/celik/Desktop/StaySafe/dataset"
    
    # Dengeli eğitim parametreleri
    batch_size = 16  # Orta seviye batch size
    num_epochs = 50  # Orta seviye epoch
    learning_rate = 0.001  # Orta seviye learning rate
    
    # CUDA optimizasyonları
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Trainer oluştur ve eğit
    trainer = FaceRecognitionTrainer(
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    print("Model eğitimi başlıyor...")
    trainer.train()
    
    print("\nTest seti üzerinde değerlendirme yapılıyor...")
    trainer.test() 