from django.shortcuts import redirect, render

# Create your views here.
def user_login(request):
    return render(request, 'account/login.html')

def user_register(request):
    return render(request, 'account/register.html')

def user_logout(request):
    return render(request, 'account/logout.html')

