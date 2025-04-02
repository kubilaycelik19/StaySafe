from django.urls import path
from . import views

app_name = 'faceRecognition_train'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train, name='train'),
    path('start-camera/', views.start_camera, name='start_camera'),
    path('stop-camera/', views.stop_camera, name='stop_camera'),
    path('start-recording/', views.start_recording, name='start_recording'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('train-model/', views.train_model, name='train_model'),
] 