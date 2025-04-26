from django.contrib import admin
from .models import Employee, Vardiya, Pozisyon

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('sicil_no', 'name', 'surname', 'age', 'vardiya')
    #list_filter = ('department',)
    search_fields = ('sicil_no', 'name', 'surname', 'vardiya')
    ordering = ('sicil_no',)

@admin.register(Vardiya)
class VardiyaAdmin(admin.ModelAdmin):
    list_display = ('vardiya_type', 'vardiya_supervizor')
    search_fields = ('vardiya_type', 'vardiya_supervizor')
    ordering = ('vardiya_type',)

@admin.register(Pozisyon)
class PozisyonAdmin(admin.ModelAdmin):
    list_display = ('pozisyon_ad',)
    search_fields = ('pozisyon_ad',)
    ordering = ('pozisyon_ad',)


