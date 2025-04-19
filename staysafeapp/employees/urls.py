from django.urls import path
from . import views

app_name = 'employees'
urlpatterns = [
    path('', views.employee_list, name='employee_list'),
    path('detail/<str:sicil_no>/', views.employee_detail, name='employee_detail'),
    path('create/', views.employee_create, name='employee_create'),
    path('edit/<str:sicil_no>/', views.employee_edit, name='employee_edit'),
    path('delete/<str:sicil_no>/', views.employee_delete, name='employee_delete'),
]

