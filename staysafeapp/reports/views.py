from django.shortcuts import render, get_object_or_404
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