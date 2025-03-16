from django.db import models

# Create your models here.

class Employee(models.Model):
    sicil_no = models.CharField(max_length=4, primary_key=True)
    name = models.CharField(max_length=100)
    surname = models.CharField(max_length=100)
    age = models.IntegerField()
    imageUrl = models.URLField(max_length=300, blank=False, default="1.jpg") # Bos birakilamaz. Denetim yapılacak.

    def __str__(self):
        return f"{self.name} {self.surname}"