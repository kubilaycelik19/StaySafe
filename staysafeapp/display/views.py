from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from static.ss_main import StaySafe
from django.views.decorators.csrf import csrf_exempt
import json

model_path = "../staysafeapp/static/Yolo11n_50_epoch.pt"
db_path = "../staysafeapp/static/Workers.db"
stay_safe = StaySafe(Model_Name=model_path, db_name=db_path)
# Kamera durumu için global değişken
camera_is_active = True

# Create your views here.
def index(request):
    return render(request, 'display/index.html', {'camera_is_active': stay_safe.camera_active})

def home(request):
    return render(request, 'display/home.html')

def video_feed(request):
    if not stay_safe.camera_active:
        return StreamingHttpResponse('')
    return StreamingHttpResponse(stay_safe.SafetyDetector(recognition=True), content_type="multipart/x-mixed-replace; boundary=frame")

@csrf_exempt
def toggle_camera(request):
    try:
        camera_is_active = stay_safe.toggle_camera()
        return JsonResponse({
            'status': 'success',
            'camera_is_active': camera_is_active
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)
