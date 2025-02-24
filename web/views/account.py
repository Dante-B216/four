from django.shortcuts import render
from web.forms.account import RegisterModelForm

def register(request):
    form = RegisterModelForm()
    return render(request, 'web/register.html', {'form': form})
