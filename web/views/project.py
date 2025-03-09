from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from django.views.decorators.csrf import csrf_exempt

from web.forms.project import ProjectModelForm

from web import models

import json
import os

from sts.sts import Sts, CIScope, Scope

from utils.tencent.cos import get_credential


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


# 取消星标
def project_delete_star(request, project_type, project_id):
    if project_type == 'my':
        models.Project.objects.filter(id=project_id, user=request.tracer).update(star=False)
        return redirect("/web/project/list")
    return HttpResponse("请求错误。")


def project_image_segmentation(request, project_id):
    context = {
        "project_id": project_id,
    }
    return render(request, "web/project_image_segmentation.html", context)


def project_manage(request, project_id):
    return render(request, "web/project_manage.html")


# 获取临时凭证
def cos_credential(request):
    data_dict = get_credential(request.tracer.bucket, request.tracer.region)
    print("data_dict", data_dict)
    return JsonResponse(data_dict)


# 将成功上传到COS的文件写入数据库
@csrf_exempt
def project_file_post(request, project_id):
    name = request.POST.get('name')
    path = request.POST.get('path')

    if not name or not path:
        return JsonResponse({'status': False, 'data': "文件错误。"})

    # 写入数据库
    instance = models.OriginalImage.objects.create(original_img_name=name, original_img_path=path,
                                                   project_id=project_id)
    result = {
        "path": instance.original_img_path
    }

    return JsonResponse({'status': True, 'data': result})


def project_handle(request, project_id):
    model_type = request.POST.get("model")
    print("model_type:", model_type)
    return JsonResponse({'status': True})
