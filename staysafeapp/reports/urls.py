from django.urls import path
from . import views

app_name = 'reports'
urlpatterns = [
    path('', views.report_list, name='report_list'),
    path('report/<int:report_id>/', views.report_detail, name='report_detail'),
    path('report/<int:report_id>/delete/', views.report_delete, name='report_delete'),
] 