import torch
import torch.nn as nn
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from torchvision import transforms
from PIL import Image
import glob
from ultralytics import YOLO

class ModelEvaluator:
    def __init__(self, data_dir='../data/test'):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_test_data(self):
        """Test verilerini yükler."""
        test_images = []
        test_labels = []
        
        # Test klasöründeki tüm resimleri tara
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_path in glob.glob(os.path.join(class_path, '*.jpg')):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.transform(img)
                        test_images.append(img_tensor)
                        test_labels.append(int(class_dir))  # Klasör adını etiket olarak kullan
                    except Exception as e:
                        print(f"Resim yüklenirken hata: {img_path} - {e}")
        
        return torch.stack(test_images), torch.tensor(test_labels)
    
    def evaluate_arcface_model(self, model_path):
        """ArcFace modelini değerlendirir."""
        try:
            # Modeli yükle
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            
            # Test verilerini yükle
            test_images, test_labels = self.load_test_data()
            test_images = test_images.to(self.device)
            
            all_preds = []
            all_labels = []
            
            # Batch işleme
            batch_size = 32
            for i in range(0, len(test_images), batch_size):
                batch_images = test_images[i:i + batch_size]
                batch_labels = test_labels[i:i + batch_size]
                
                with torch.no_grad():
                    outputs = model(batch_images)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.numpy())
            
            # Metrikleri hesapla
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            return accuracy, f1
            
        except Exception as e:
            print(f"ArcFace modeli değerlendirilirken hata: {e}")
            return None, None
    
    def evaluate_yolo_model(self, model_path):
        """YOLO modelini değerlendirir."""
        try:
            # YOLO modelini yükle
            model = YOLO(model_path)
            
            # Test verilerini yükle
            test_images, test_labels = self.load_test_data()
            
            # Ultralytics YOLO modeli ile değerlendirme
            results = model.val(data=self.data_dir, split='test', imgsz=640, batch=32)
            
            # Metrikleri al
            metrics = results.results_dict
            mAP50 = metrics.get('metrics/mAP50', 0.0)
            mAP50_95 = metrics.get('metrics/mAP50-95', 0.0)
            precision = metrics.get('metrics/precision', 0.0)
            recall = metrics.get('metrics/recall', 0.0)
            
            # IoU değerlerini al (farklı eşikler için)
            iou_values = {
                'IoU@0.5': metrics.get('metrics/IoU@0.5', 0.0),
                'IoU@0.75': metrics.get('metrics/IoU@0.75', 0.0),
                'IoU@0.9': metrics.get('metrics/IoU@0.9', 0.0)
            }
            
            return {
                'mAP50': mAP50,
                'mAP50_95': mAP50_95,
                'precision': precision,
                'recall': recall,
                'iou_values': iou_values
            }
            
        except Exception as e:
            print(f"YOLO modeli değerlendirilirken hata: {e}")
            return None

class PerformanceVisualizer:
    def __init__(self, data_file='model_performance.json'):
        self.data_file = data_file
        self.data = self._load_data()
    
    def _load_data(self):
        """JSON dosyasından model performans verilerini yükler."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'arcface': [], 'yolo': []}
    
    def add_model_performance(self, model_name, model_type, metrics, model_path):
        """Yeni model performans verisi ekler."""
        if model_type not in ['arcface', 'yolo']:
            raise ValueError("model_type 'arcface' veya 'yolo' olmalıdır")
        
        performance_data = {
            'model_name': model_name,
            'model_path': model_path,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if model_type == 'arcface':
            performance_data.update({
                'accuracy': metrics[0],
                'f1_score': metrics[1]
            })
        else:  # yolo
            performance_data.update(metrics)
        
        self.data[model_type].append(performance_data)
        self._save_data()
    
    def _save_data(self):
        """Verileri JSON dosyasına kaydeder."""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
    
    def visualize_performance(self, save_path=None):
        """Model performans verilerini görselleştirir."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # ArcFace modelleri için grafik
        if self.data['arcface']:
            for model in self.data['arcface']:
                model_name = model['model_name']
                
                # Accuracy grafiği
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(['Accuracy'], [model['accuracy']], color='skyblue')
                ax.set_title(f'ArcFace Model Accuracy - {model_name}')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Değeri çubuğun üzerine yaz
                ax.text(0, model['accuracy'] + 0.02, f'{model["accuracy"]:.4f}', 
                       ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(f'{model_name}_accuracy_{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
                
                # F1 Score grafiği
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(['F1 Score'], [model['f1_score']], color='lightcoral')
                ax.set_title(f'ArcFace Model F1 Score - {model_name}')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Değeri çubuğun üzerine yaz
                ax.text(0, model['f1_score'] + 0.02, f'{model["f1_score"]:.4f}', 
                       ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(f'{model_name}_f1score_{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
        
        # YOLO modelleri için grafik
        if self.data['yolo']:
            for model in self.data['yolo']:
                model_name = model['model_name']
                
                # mAP değerleri grafiği
                fig, ax = plt.subplots(figsize=(10, 6))
                x = ['mAP50', 'mAP50-95']
                values = [model['mAP50'], model['mAP50_95']]
                bars = ax.bar(x, values, color=['skyblue', 'lightcoral'])
                ax.set_title(f'YOLO Model mAP Values - {model_name}')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Değerleri çubukların üzerine yaz
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
                           f'{value:.4f}', ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(f'{model_name}_map_{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Precision-Recall grafiği
                fig, ax = plt.subplots(figsize=(10, 6))
                x = ['Precision', 'Recall']
                values = [model['precision'], model['recall']]
                bars = ax.bar(x, values, color=['skyblue', 'lightcoral'])
                ax.set_title(f'YOLO Model Precision-Recall - {model_name}')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Değerleri çubukların üzerine yaz
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
                           f'{value:.4f}', ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(f'{model_name}_precision_recall_{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
                
                # IoU değerleri grafiği
                fig, ax = plt.subplots(figsize=(10, 6))
                x = ['IoU@0.5', 'IoU@0.75', 'IoU@0.9']
                values = [model['iou_values']['IoU@0.5'], 
                         model['iou_values']['IoU@0.75'],
                         model['iou_values']['IoU@0.9']]
                bars = ax.bar(x, values, color=['skyblue', 'lightcoral', 'lightgreen'])
                ax.set_title(f'YOLO Model IoU Values - {model_name}')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Değerleri çubukların üzerine yaz
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
                           f'{value:.4f}', ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(f'{model_name}_iou_{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Tüm metriklerin tablosu
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.axis('tight')
                ax.axis('off')
                
                table_data = [
                    ['Metric', 'Value'],
                    ['mAP50', f"{model['mAP50']:.4f}"],
                    ['mAP50-95', f"{model['mAP50_95']:.4f}"],
                    ['Precision', f"{model['precision']:.4f}"],
                    ['Recall', f"{model['recall']:.4f}"],
                    ['IoU@0.5', f"{model['iou_values']['IoU@0.5']:.4f}"],
                    ['IoU@0.75', f"{model['iou_values']['IoU@0.75']:.4f}"],
                    ['IoU@0.9', f"{model['iou_values']['IoU@0.9']:.4f}"]
                ]
                
                table = ax.table(
                    cellText=table_data,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.4, 0.4]
                )
                
                # Tablo başlığını ayarla
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 2)
                
                # Başlık hücrelerini vurgula
                for i in range(2):
                    table[(0, i)].set_facecolor('#40466e')
                    table[(0, i)].set_text_props(color='white', weight='bold')
                
                plt.title(f'YOLO Model Metrics Summary - {model_name}', pad=20)
                plt.tight_layout()
                if save_path:
                    plt.savefig(f'{model_name}_metrics_summary_{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
    
    def print_performance_table(self):
        """Model performans verilerini tablo formatında yazdırır."""
        print("\n=== ArcFace Modelleri ===")
        print(f"{'Model Adı':<30} {'Accuracy':<10} {'F1 Score':<10} {'Oluşturulma Tarihi':<20}")
        print("-" * 70)
        for model in self.data['arcface']:
            print(f"{model['model_name']:<30} {model['accuracy']:<10.2f} {model['f1_score']:<10.2f} {model['created_at']:<20}")
        
        print("\n=== YOLO Modelleri ===")
        print(f"{'Model Adı':<30} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'IoU@0.5':<10} {'IoU@0.75':<10} {'IoU@0.9':<10}")
        print("-" * 100)
        for model in self.data['yolo']:
            print(f"{model['model_name']:<30} "
                  f"{model['mAP50']:<10.3f} "
                  f"{model['mAP50_95']:<10.3f} "
                  f"{model['precision']:<10.3f} "
                  f"{model['recall']:<10.3f} "
                  f"{model['iou_values']['IoU@0.5']:<10.3f} "
                  f"{model['iou_values']['IoU@0.75']:<10.3f} "
                  f"{model['iou_values']['IoU@0.9']:<10.3f}")

def main():
    # Model değerlendirici ve görselleştirici oluştur
    evaluator = ModelEvaluator()
    visualizer = PerformanceVisualizer()
    
    # ArcFace modelini değerlendir
    arcface_model_path = 'static/models/best_epoch1_lr0.001_acc0.9273.pth'
    if os.path.exists(arcface_model_path):
        print(f"\nArcFace modeli değerlendiriliyor: {arcface_model_path}")
        metrics = evaluator.evaluate_arcface_model(arcface_model_path)
        
        if metrics is not None:
            model_name = os.path.splitext(os.path.basename(arcface_model_path))[0]
            visualizer.add_model_performance(
                model_name=model_name,
                model_type='arcface',
                metrics=metrics,
                model_path=arcface_model_path
            )
    
    # YOLO modelini değerlendir
    yolo_model_path = 'static/Yolo11n_50_epoch.pt'
    if os.path.exists(yolo_model_path):
        print(f"\nYOLO modeli değerlendiriliyor: {yolo_model_path}")
        metrics = evaluator.evaluate_yolo_model(yolo_model_path)
        
        if metrics is not None:
            model_name = os.path.splitext(os.path.basename(yolo_model_path))[0]
            visualizer.add_model_performance(
                model_name=model_name,
                model_type='yolo',
                metrics=metrics,
                model_path=yolo_model_path
            )
    
    # Performans tablosunu yazdır
    visualizer.print_performance_table()
    
    # Grafikleri göster ve kaydet
    visualizer.visualize_performance(save_path='model_performance.png')

if __name__ == "__main__":
    main()
