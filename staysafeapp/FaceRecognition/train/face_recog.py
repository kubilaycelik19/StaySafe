import torch
import torch.nn as nn

class LivenessModel(nn.Module):
    def __init__(self):
        super(LivenessModel, self).__init__()
        
        # 3D Convolutional katmanları
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        
        # MaxPooling katmanları
        self.pool = nn.MaxPool3d(kernel_size=2)
        
        # Dropout katmanları
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Dense katmanları
        self.flatten = nn.Flatten()
        # Giriş boyutunu düzelt: 64 kanal * 1 * 10 * 10 = 6400
        self.fc1 = nn.Linear(6400, 128)
        self.fc2 = nn.Linear(128, 2)
        
        # Aktivasyon fonksiyonları
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # İlk konvolüsyon bloğu
        x = self.relu(self.conv1(x))
        
        # İkinci konvolüsyon bloğu
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Üçüncü konvolüsyon bloğu
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Dördüncü konvolüsyon bloğu
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        # Dropout ve Flatten
        x = self.dropout1(x)
        x = self.flatten(x)
        
        # Dense katmanları
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

def get_liveness_model():
    model = LivenessModel()
    return model 