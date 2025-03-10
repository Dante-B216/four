from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from django.views.decorators.csrf import csrf_exempt

from web.forms.project import ProjectModelForm

from web import models

from django.urls import reverse

import json
import os

from sts.sts import Sts, CIScope, Scope

from utils.tencent.cos import get_credential

import requests


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
    key = request.POST.get('key')

    if not name or not path:
        return JsonResponse({'status': False, 'data': "文件错误。"})

    # 写入数据库
    instance = models.OriginalImage.objects.create(original_img_name=name, original_img_path=path, original_img_key=key,
                                                   project_id=project_id)

    # 把数据传回给前端
    result = {
        "original_img_id": instance.id,
        "original_img_path": instance.original_img_path,
        "download_url": reverse("web:file_download", kwargs={'project_id': project_id, 'original_img_id': instance.id})
    }

    print("result", result)

    return JsonResponse({'status': True, 'data': result})


# 图像分割
def project_handle(request, project_id):
    model_type = request.POST.get("model")
    print("model_type:", model_type)
    return JsonResponse({'status': True})


# 下载文件
def file_download(request, project_id, original_img_id):

    # 获取要下载的文件对象
    file_object = models.OriginalImage.objects.filter(id=original_img_id, project_id=project_id).first()

    # 获取要下载的文件对象的COS路径
    res = requests.get(file_object.original_img_path)

    data = res.content

    response = HttpResponse(data)

    # 设置响应头
    response["Content-Disposition"] = "attachment; filename={}".format(file_object.original_img_key)

    return response
