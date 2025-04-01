from django.db import models
from employees.models import Employee

class EmployeeReport(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='reports')
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='reports/')
    is_equipped = models.BooleanField(default=False)
    location = models.CharField(max_length=200, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Çalışan Raporu'
        verbose_name_plural = 'Çalışan Raporları'

    def __str__(self):
        return f"{self.employee.name} {self.employee.surname} - {self.timestamp}" 