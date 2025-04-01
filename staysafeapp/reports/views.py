from django.shortcuts import render

def report_list(request):
    return render(request, 'reports/report_list.html')

def report_detail(request, report_id):
    return render(request, 'reports/report_detail.html')