from django.contrib import admin
from .models import Employee

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('sicil_no', 'name', 'surname', 'age')
    #list_filter = ('department',)
    search_fields = ('sicil_no', 'name', 'surname')
    ordering = ('sicil_no',)

