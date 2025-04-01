from django.urls import path
from . import views

urlpatterns = [
    path('', views.report_list, name='report_list'),
    path('detail/<int:report_id>/', views.report_detail, name='report_detail'),
] 