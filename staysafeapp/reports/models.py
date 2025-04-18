from django.db import models
from employees.models import Employee

class EmployeeReport(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.SET_NULL, related_name='reports', null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='reports/')
    is_equipped = models.BooleanField(default=False)
    location = models.CharField(max_length=200, blank=True)
    notes = models.TextField(blank=True)
    missing_equipment = models.TextField(blank=True, verbose_name="Eksik Ekipmanlar")

    # Rapor anındaki durumu kaydetmek için eklenen denormalize alanlar
    reported_pozisyon = models.CharField(max_length=100, null=True, blank=True, verbose_name="Rapor Anındaki Pozisyon")
    reported_vardiya = models.CharField(max_length=50, null=True, blank=True, verbose_name="Rapor Anındaki Vardiya")
    reported_supervizor_name = models.CharField(max_length=200, null=True, blank=True, verbose_name="Rapor Anındaki Süpervizör")

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Çalışan Raporu'
        verbose_name_plural = 'Çalışan Raporları'

    def __str__(self):
        # Tarih ve saat formatını gün/ay/yıl saat:dakika:saniye olarak ayarla
        formatted_timestamp = self.timestamp.strftime('%d/%m/%Y %H:%M:%S')
        if self.employee:
            return f"{self.employee.name} {self.employee.surname} - {formatted_timestamp}"
        else:
            return f"Bilinmeyen Çalışan Raporu - {formatted_timestamp}" 