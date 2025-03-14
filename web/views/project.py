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

from utils.tencent.cos import download_file
from utils.img import segment_image
from utils.tencent.cos import upload_file
from utils.tencent.cos import delete_file
from utils.tencent.cos import delete_bucket
from utils import encrypt
from django.contrib import messages


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
def project_list(request, user_id):
    if request.method == 'GET':
        project_dict = {'star': [], 'my': []}

        # 我创建的所有项目
        my_project_list = models.Project.objects.filter(user_id=user_id)

        # 原图和结果图都存在的项目
        data_list = []

        # 仅展示原图和结果图都存在的项目
        for project_object in my_project_list:
            original_img_object = models.OriginalImage.objects.filter(project_id=project_object.id).first()
            if original_img_object:
                result_img_object = models.SegmentationResult.objects.filter(
                    original_img_id=original_img_object.id).first()
                if result_img_object:
                    data_list.append(project_object)

        for row in data_list:
            if row.star:
                project_dict['star'].append(row)
            else:
                project_dict['my'].append(row)

        return render(request, 'web/project_list.html', {'project_dict': project_dict})


# 星标项目
def project_star(request, project_type, project_id):
    if project_type == 'my':
        models.Project.objects.filter(id=project_id, user=request.tracer).update(star=True)
        url = reverse('web:project_list', kwargs={'user_id': request.tracer.id})
        return redirect(url)
    return HttpResponse("请求错误。")


# 取消星标
def project_delete_star(request, project_type, project_id):
    if project_type == 'my':
        models.Project.objects.filter(id=project_id, user=request.tracer).update(star=False)
        url = reverse('web:project_list', kwargs={'user_id': request.tracer.id})
        return redirect(url)
    return HttpResponse("请求错误。")


def project_image_segmentation(request, project_id):
    context = {
        "project_id": project_id,
    }
    return render(request, "web/project_image_segmentation.html", context)


# 图像分割项目展示
def project_manage(request, project_id):
    # 获取项目对象
    project_object = models.Project.objects.filter(id=project_id, user_id=request.tracer.id).first()

    # 获取上传图像对象
    original_img_object = models.OriginalImage.objects.filter(project_id=project_id).first()

    # 获取结果图像对象
    result_img_object = models.SegmentationResult.objects.filter(original_img_id=original_img_object.id).first()

    # 获取项目信息
    project_id = project_object.id
    project_name = project_object.name
    project_description = project_object.description
    project_time = project_object.project_time

    # 获取上传图片信息
    original_img_id = original_img_object.id
    original_img_path = original_img_object.original_img_path
    original_img_time = original_img_object.original_img_time

    # 获取结果图片信息
    result_img_id = result_img_object.id
    model_type = result_img_object.get_model_type_display
    result_img_path = result_img_object.result_img_path
    result_img_time = result_img_object.result_img_time

    context = {
        "project_id": project_id,
        "project_name": project_name,
        "project_description": project_description,
        "project_time": project_time,
        "original_img_id": original_img_id,
        "original_img_path": original_img_path,
        "original_img_time": original_img_time,
        "result_img_id": result_img_id,
        "model_type": model_type,
        "result_img_path": result_img_path,
        "result_img_time": result_img_time,
    }

    print(context)

    return render(request, "web/project_manage.html", context)


# 获取临时凭证
def cos_credential(request):
    data_dict = get_credential(request.tracer.bucket, request.tracer.region)
    print("data_dict", data_dict)
    return JsonResponse(data_dict)


# 将前端成功上传到COS的文件写入数据库
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
        "download_url": reverse("web:original_img_file_download",
                                kwargs={'project_id': project_id, 'original_img_id': instance.id})
    }

    print("result", result)

    return JsonResponse({'status': True, 'data': result})


# 图像分割
def project_handle(request, project_id, original_img_id):
    model_type = request.POST.get("model")
    print("model_type:", model_type)

    if model_type == '1':
        model_path = "ml/netModels/unet.pth"
    elif model_type == '2':
        model_path = "ml/netModels/unet_c.pth"
    elif model_type == '3':
        model_path = "ml/netModels/unet_s.pth"
    elif model_type == '4':
        model_path = "ml/netModels/unet_cs.pth"
    elif model_type == '5':
        model_path = "ml/netModels/unet++.pth"
    elif model_type == '6':
        model_path = "ml/netModels/u2net.pth"

    print("model_path", model_path)

    # 获取要分割的图片对象
    original_img = models.OriginalImage.objects.filter(id=original_img_id, project_id=project_id).first()

    # 获取要分割的图片COS名称
    original_img_key = original_img.original_img_key
    print("original_img_path", original_img_key)

    # 把图片下载到本地
    download_file(request.tracer.bucket, request.tracer.region, original_img_key)

    # 图片本地路径
    original_img_local_path = "web/views/handle_img/" + original_img_key
    print("original_img_local_path", original_img_local_path)

    # 分割结果路径
    result_img_local_path = segment_image(original_img_local_path, model_path)
    print("result_img_local_path", result_img_local_path)

    # 获取分割结果文件名
    result_img_key = os.path.basename(result_img_local_path)

    # 重点：用 with open 打开文件，传递文件对象给 Body
    with open(result_img_local_path, "rb") as file_object:  # 二进制模式读取
        result_img_path = upload_file(request.tracer.bucket, request.tracer.region, file_object,
                                      result_img_key)
        print("result_img_path", result_img_path)

    # 写入数据库
    instance = models.SegmentationResult.objects.create(model_type=model_type, result_img_path=result_img_path,
                                                        result_img_key=result_img_key,
                                                        original_img_id=original_img_id)

    # 删除本地文件
    os.remove(result_img_local_path)
    os.remove(original_img_local_path)

    # 把数据传回给前端
    result = {
        "result_img_id": instance.id,
        "result_img_path": instance.result_img_path,
        "download_url": reverse("web:result_img_file_download",
                                kwargs={'project_id': project_id, 'original_img_id': original_img_id,
                                        'result_img_id': instance.id})
    }

    print("result", result)

    return JsonResponse({'status': True, 'data': result})


# 下载上传文件
def original_img_file_download(request, project_id, original_img_id):
    # 获取要下载的文件对象
    file_object = models.OriginalImage.objects.filter(id=original_img_id, project_id=project_id).first()

    # 获取要下载的文件对象的COS路径
    res = requests.get(file_object.original_img_path)

    data = res.content

    response = HttpResponse(data)

    # 设置响应头
    response["Content-Disposition"] = "attachment; filename={}".format(file_object.original_img_key)

    return response


# 下载结果文件
def result_img_file_download(request, project_id, original_img_id, result_img_id):
    # 获取要下载的文件对象
    file_object = models.SegmentationResult.objects.filter(id=result_img_id, original_img_id=original_img_id).first()

    # 获取要下载的文件对象的COS路径
    res = requests.get(file_object.result_img_path)

    data = res.content

    response = HttpResponse(data)

    # 设置响应头
    response["Content-Disposition"] = "attachment; filename={}".format(file_object.result_img_key)

    return response


# 删除项目
def project_delete(request, project_id):
    # 获取原图
    original_img_object = models.OriginalImage.objects.filter(project_id=project_id).first()
    original_img_id = original_img_object.id
    original_img_key = original_img_object.original_img_key

    # 获取结果图
    result_img_object = models.SegmentationResult.objects.filter(original_img_id=original_img_id).first()
    result_img_key = result_img_object.result_img_key

    # 删除COS图片
    original_img_result = delete_file(request.tracer.bucket, request.tracer.region, original_img_key)

    result_img_result = delete_file(request.tracer.bucket, request.tracer.region, result_img_key)

    print("original_img_result", original_img_result)
    print("result_img_result", result_img_result)

    if original_img_result != None and result_img_result != None:
        models.Project.objects.filter(id=project_id, user_id=request.tracer.id).delete()

        url = reverse('web:project_list', kwargs={'user_id': request.tracer.id})  # 跳转回项目列表

    else:
        url = reverse('web:project_manage', kwargs={'project_id': project_id})  # 跳转回项目列表

    return redirect(url)


# 个人中心
def personal_center(request, user_id):
    user_object = models.UserInfo.objects.filter(id=user_id).first()
    user_name = user_object.user_name
    user_email = user_object.user_email
    user_phone = user_object.user_phone
    context = {
        'user_id': user_id,
        'user_name': user_name,
        'user_email': user_email,
        'user_phone': user_phone,
    }
    return render(request, "web/personal_center.html", context)


# 注销用户
def personal_center_delete(request, user_id):
    if request.method == 'GET':
        return render(request, "web/personal_center_delete.html")

    if request.method == 'POST':
        user_name = request.POST.get('user_name')
        if not user_name or user_name != request.tracer.user_name:
            return render(request, "web/personal_center_delete.html", {"error": "用户名错误。"})

        delete_bucket(request.tracer.bucket, request.tracer.region)

        models.UserInfo.objects.filter(id=request.tracer.id).delete()

        request.session.flush()

        return redirect("web:index")


# 修改密码
def personal_center_change_pwd(request, user_id):
    if request.method == 'GET':
        return render(request, "web/personal_center_change_pwd.html")

    if request.method == 'POST':
        error = None
        success = False

        # 获取当前用户对象
        user = models.UserInfo.objects.filter(id=request.tracer.id).first()

        if not user:
            return redirect('web:login_name')

        # 获取表单数据
        old_password = request.POST.get('old_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        # 验证原密码
        if encrypt.md5(old_password) != user.user_pw:
            error = "原密码错误。"
        else:
            # 前端验证规则（与注册一致）
            if new_password != confirm_password:
                error = "两次输入的新密码不一致。"
            elif not (8 <= len(new_password) <= 32):
                error = "密码长度需在8-32位之间。"
            elif not any(char.isalpha() for char in new_password) or not any(char.isdigit() for char in new_password):
                error = "密码必须包含字母和数字。"
            else:
                # 更新密码
                user.user_pw = encrypt.md5(new_password)
                user.save()
                success = True
        return render(request, "web/personal_center_change_pwd.html", {
            'error': error,
            'success': success,
            'user': user
        })
