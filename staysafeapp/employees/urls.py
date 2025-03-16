from django.urls import path
from . import views

urlpatterns = [
    path('index', views.index, name='index'),
    path('detail/<str:sicil_no>', views.employee_detail, name='employee_detail'),
]

