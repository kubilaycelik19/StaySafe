from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST # Sadece POST isteği için
from django.contrib import messages # Kullanıcıya mesaj göstermek için
from .models import EmployeeReport

def report_list(request):
    """
    Veritabanındaki tüm güvenlik raporlarını listeler.
    """
    reports = EmployeeReport.objects.all() # Tüm raporları al (ordering model meta'da tanımlı)
    context = {
        'reports': reports
    }
    return render(request, 'reports/report_list.html', context)

def report_detail(request, report_id):
    """Belirli bir raporun detaylarını gösterir."""
    report = get_object_or_404(EmployeeReport, pk=report_id) # Raporu ID ile al veya 404 döndür
    context = {
        'report': report
    }
    return render(request, 'reports/report_detail.html', context)

@require_POST # Bu view sadece POST isteklerini kabul etsin
# @login_required # Kimlik doğrulama eklendiğinde bu satır aktif edilecek
# @permission_required('reports.delete_employeereport', raise_exception=True) # Yetkilendirme eklendiğinde
def report_delete(request, report_id):
    """Belirli bir raporu siler."""
    report = get_object_or_404(EmployeeReport, pk=report_id)
    try:
        # İlişkili görsel dosyasını sil (eğer varsa)
        if report.image:
            report.image.delete(save=False) # save=False veritabanını tekrar güncellemez

        report.delete() # Raporu veritabanından sil
        messages.success(request, f"Rapor (ID: {report_id}) başarıyla silindi.")
    except Exception as e:
        messages.error(request, f"Rapor silinirken bir hata oluştu: {e}")

    return redirect('reports:report_list') # Rapor listesine geri yönlendir