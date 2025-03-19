from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from static.ss_main import StaySafe

stay_safe = StaySafe(Model_Name="../staysafeapp/static/Yolo11n_50_epoch.pt", db_name="staysafeapp/static/Workers.db")

label_to_name = {
    0: "Emre",  # 0 etiketinin ismi
    1: "Kubilay"    # 1 etiketinin ismi
}

# Kamera durumu için global değişken
camera_is_active = True

# Create your views here.
def index(request):
    return render(request, 'display/index.html', {'camera_is_active': camera_is_active})

def home(request):
    return render(request, 'display/home.html')

def video_feed(request):
    if not camera_is_active:
        return StreamingHttpResponse('')
    return StreamingHttpResponse(stay_safe.SafetyDetector(recognition=True), content_type="multipart/x-mixed-replace; boundary=frame")

def toggle_camera(request):
    global camera_is_active
    camera_is_active = not camera_is_active
    return JsonResponse({'status': 'success', 'camera_is_active': camera_is_active})
