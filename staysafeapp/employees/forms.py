from django import forms
from .models import Employee

class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['sicil_no', 'name', 'surname', 'age']
        widgets = {
            'sicil_no': forms.TextInput(attrs={'class': 'form-control'}),
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'surname': forms.TextInput(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control'}),
            #'imageUrl': forms.URLInput(attrs={'class': 'form-control'}), # Aktif edildiğinde fields'a eklenecek.
        } 