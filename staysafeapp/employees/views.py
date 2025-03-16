from django.shortcuts import render, get_object_or_404
from .models import Employee

# Create your views here.

def index(request):
    employees = Employee.objects.all().order_by('sicil_no')
    return render(request, 'employees/index.html', {
        'employees': employees
    })

def employee_detail(request, sicil_no):
    employee = get_object_or_404(Employee, sicil_no=sicil_no)
    return render(request, 'employees/employee_detail.html', {
        'employee': employee
    })
