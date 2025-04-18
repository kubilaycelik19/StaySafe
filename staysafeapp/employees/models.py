from django.db import models

# Create your models here.

class Pozisyon(models.Model):
    pozisyon_ad = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.pozisyon_ad

class Vardiya(models.Model):
    GECE = 'GECE'
    GUNDUZ = 'GUNDUZ'
    VARDIYA_TURU_SECENEKLERI = [
        (GECE, 'Gece'),
        (GUNDUZ, 'Gündüz'),
    ]
    vardiya_type = models.CharField(
        max_length=6,
        choices=VARDIYA_TURU_SECENEKLERI,
        default=GUNDUZ,
    )
    vardiya_supervizor = models.ForeignKey(
        'Employee',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='supervised_shifts'
    )

    def __str__(self):
        return self.get_vardiya_type_display()

class Employee(models.Model):
    sicil_no = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=100)
    surname = models.CharField(max_length=100)
    age = models.IntegerField(null=True, blank=True)
    imageUrl = models.URLField(max_length=300, blank=True, null=True, default="default.jpg")
    pozisyon = models.ForeignKey(Pozisyon, on_delete=models.SET_NULL, null=True, blank=True)
    vardiya = models.ForeignKey(Vardiya, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.name} {self.surname} ({self.sicil_no})"