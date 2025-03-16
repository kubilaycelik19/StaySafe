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
    def __init__(self, data_dir, batch_size=32, num_epochs=50, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Çok daha zorlu veri dönüşümleri
        self.train_transform = transforms.Compose([
            transforms.Resize((320, 320)),  # Daha büyük boyuta getir
            transforms.RandomCrop(224),     # Daha agresif kırpma
            transforms.RandomHorizontalFlip(p=0.7),  # Flip olasılığı artırıldı
            transforms.RandomRotation(30),  # Daha fazla rotasyon
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # Daha agresif silme
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation için de zorlu dönüşümler
        self.val_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(224),  # Validation'da da random crop
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.3),  # Validation'da da flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Veri yükleyicileri
        self.train_dataset = FaceDataset(os.path.join(data_dir, 'train'), self.train_transform)
        self.val_dataset = FaceDataset(os.path.join(data_dir, 'val'), self.val_transform)
        self.test_dataset = FaceDataset(os.path.join(data_dir, 'test'), self.val_transform)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # Çoklu işçi eklendi
            pin_memory=True  # CUDA için optimizasyon
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        
        # Model ve loss fonksiyonu
        self.num_classes = len(self.train_dataset.classes)
        self.model = self.create_model()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # Label smoothing artırıldı
        
        # Optimizer ve scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.1,  # L2 regularization artırıldı
            betas=(0.9, 0.99),
            eps=1e-8
        )
        
        # Daha agresif learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        # Eğitim geçmişi
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def create_model(self):
        # Daha basit model mimarisi
        model = models.resnet18(weights=None)  # Pretrained weights kullanma
        
        # Convolutional katmanlarına Dropout ekle
        model.layer1[0].conv1 = nn.Sequential(
            model.layer1[0].conv1,
            nn.Dropout2d(0.2)
        )
        model.layer2[0].conv1 = nn.Sequential(
            model.layer2[0].conv1,
            nn.Dropout2d(0.3)
        )
        model.layer3[0].conv1 = nn.Sequential(
            model.layer3[0].conv1,
            nn.Dropout2d(0.4)
        )
        model.layer4[0].conv1 = nn.Sequential(
            model.layer4[0].conv1,
            nn.Dropout2d(0.5)
        )
        
        # Basitleştirilmiş FC katmanları
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, self.num_classes)
        )
        
        # He initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        return model.to(self.device)
    
    def train(self):
        print("\nEğitim başlıyor...")
        print("Model yapılandırması:")
        print("- Model: ResNet18 (pretrained weights yok)")
        print("- Giriş boyutu: 224x224")
        print(f"- Sınıf sayısı: {self.num_classes}")
        print("- Dropout2d: 0.2->0.3->0.4->0.5")
        print("- FC Dropout: 0.5")
        print("- Weight decay: 0.1")
        print("- Label smoothing: 0.2")
        
        best_val_acc = 0.0
        patience = 20  # Patience azaltıldı
        patience_counter = 0
        
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
                
                # Gradient clipping ekle
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': f'{train_loss/len(self.train_loader):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
            
            # Doğrulama
            val_loss, val_acc = self.validate()
            
            # Early stopping ve model kaydetme
            if val_acc < 95.0 and val_acc > best_val_acc:  # Threshold düşürüldü
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model()
                print(f'\nYeni en iyi model! (Val Acc: {val_acc:.2f}%)')
                print(f'Learning rate: {self.scheduler.get_last_lr()[0]:.6f}')
            else:
                patience_counter += 1
                if val_acc >= 95.0:
                    print(f'\nValidation accuracy çok yüksek - muhtemel overfitting, model kaydedilmedi')
                print(f'\nİyileşme yok. Patience: {patience_counter}/{patience}')
                if patience_counter >= patience:
                    print(f'\nEarly stopping! En iyi validation accuracy: {best_val_acc:.2f}%')
                    break
            
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
    
    # Eğitim parametreleri
    batch_size = 4  # Batch size daha da küçültüldü
    num_epochs = 100  # Epoch sayısı artırıldı
    learning_rate = 0.001  # Learning rate artırıldı
    
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