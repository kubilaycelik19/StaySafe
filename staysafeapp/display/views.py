from django.http import StreamingHttpResponse
from django.shortcuts import render
from static.staysafe_main import StaySafe

stay_safe = StaySafe(Model_Name="../staysafeapp/static/Yolo11n_50_epoch.pt", face_model_path="../staysafeapp/static/emre_kubilay_100_epoch_resnet.pth", db_name="staysafeapp/static/Workers.db")

label_to_name = {
    0: "Emre",  # 0 etiketinin ismi
    1: "Kubilay"    # 1 etiketinin ismi
}

# Create your views here.
def index(request):
    return render(request, 'display/index.html')

def video_feed(request):
    return StreamingHttpResponse(stay_safe.SafetyDetector(Source=0, recognition=True), content_type="multipart/x-mixed-replace; boundary=frame")
