from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from web.forms.project import ProjectModelForm

from web import models


def project_add(request):
    if request.method == 'GET':
        form = ProjectModelForm(request)
        return render(request, 'web/project_add.html', {'form': form})

    form = ProjectModelForm(request, data=request.POST)
    if form.is_valid():
        print("form is_valid")
        print(form.cleaned_data)
        form.instance.user = request.tracer
        form.save()
        url = f'/web/project/image_segmentation/{form.instance.id}/'
        return JsonResponse({'status': True, 'data': url})
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({'status': False, 'error': form.errors})


# 展示项目
def project_list(request):
    if request.method == 'GET':
        project_dict = {'star': [], 'my': []}

        my_project_list = models.Project.objects.filter(user=request.tracer)

        for row in my_project_list:
            if row.star:
                project_dict['star'].append(row)
            else:
                project_dict['my'].append(row)

        return render(request, 'web/project_list.html', {'project_dict': project_dict})


# 星标项目
def project_star(request, project_type, project_id):
    if project_type == 'my':
        models.Project.objects.filter(id=project_id, user=request.tracer).update(star=True)
        return redirect("/web/project/list")
    return HttpResponse("请求错误。")


def project_delete_star(request, project_type, project_id):
    if project_type == 'my':
        models.Project.objects.filter(id=project_id, user=request.tracer).update(star=False)
        return redirect("/web/project/list")
    return HttpResponse("请求错误。")


def project_image_segmentation(request, project_id):
    return render(request, "web/project_image_segmentation.html")


def project_manage(request, project_id):
    return render(request, "web/project_manage.html")
