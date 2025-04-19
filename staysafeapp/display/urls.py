from django.urls import path
from . import views

app_name = 'display'
urlpatterns = [
    path('', views.home, name='home'),
    path('index/', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('toggle_camera/', views.toggle_camera, name='toggle_camera'),
]
