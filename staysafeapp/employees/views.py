from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import Employee
from .forms import EmployeeForm

# Create your views here.

def index(request):
    employees = Employee.objects.all().order_by('sicil_no')
    return render(request, 'employees/index.html', {
        'employees': employees
    })

def employee_list(request):
    employees = Employee.objects.all().order_by('sicil_no')
    return render(request, 'employees/index.html', {'employees': employees})

def employee_detail(request, sicil_no):
    employee = get_object_or_404(Employee, sicil_no=sicil_no)
    return render(request, 'employees/employee_detail.html', {
        'employee': employee
    })

def employee_create(request):
    if request.method == 'POST':
        form = EmployeeForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Çalışan başarıyla eklendi.')
            return redirect('employee_list')
    else:
        form = EmployeeForm()
    return render(request, 'employees/employee_form.html', {'form': form})

def employee_edit(request, sicil_no):
    employee = get_object_or_404(Employee, sicil_no=sicil_no)
    if request.method == 'POST':
        form = EmployeeForm(request.POST, instance=employee)
        if form.is_valid():
            form.save()
            messages.success(request, 'Çalışan başarıyla güncellendi.')
            return redirect('employee_list')
    else:
        form = EmployeeForm(instance=employee)
    return render(request, 'employees/employee_form.html', {'form': form})

def employee_delete(request, sicil_no):
    employee = get_object_or_404(Employee, sicil_no=sicil_no)
    if request.method == 'POST':
        employee.delete()
        messages.success(request, 'Çalışan başarıyla silindi.')
        return redirect('employee_list')
    return render(request, 'employees/employee_confirm_delete.html', {'employee': employee})
