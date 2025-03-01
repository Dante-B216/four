from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from web.forms.project import ProjectModelForm


def project_add(request):
    if request.method == 'GET':
        form = ProjectModelForm()
        return render(request, 'web/project_add.html', {'form': form})


    form = ProjectModelForm(data=request.POST)
    if form.is_valid():
        print("form is_valid")
        print(form.cleaned_data)
        form.instance.user = request.tracer
        form.save()
        return JsonResponse({'status': True})
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({'status': False, 'error': form.errors})



def project_list(request):
    if request.method == 'GET':
        return render(request, 'web/project_list.html')
