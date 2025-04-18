from django.db import models
from employees.models import Employee # Employee modelini import et

# Create your models here.

class Camera(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    is_active = models.BooleanField(default=True)
    # last_frame için ImageField kullanıldı, MEDIA_ROOT ve MEDIA_URL ayarları gerekli olabilir.
    last_frame = models.ImageField(upload_to='camera_frames/', blank=True, null=True)
    last_update = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.location})"

class DetectionLog(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_equipped = models.BooleanField(default=False) # Ekipman takılı mı?
    # frame için ImageField kullanıldı
    frame = models.ImageField(upload_to='detection_frames/')
    confidence = models.FloatField(default=0.0) # Tespit güven skoru

    class Meta:
        ordering = ['-timestamp'] # Kayıtları en yeniden eskiye sırala

    def __str__(self):
        status = "Ekipmanli" if self.is_equipped else "Ekipmansiz"
        return f"{self.employee} - {self.camera.name} - {status} ({self.timestamp.strftime('%Y-%m-%d %H:%M')})"
