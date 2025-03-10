from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from web import models
from django.urls import reverse

from web.forms.wiki import WikiAddModelForm
from web.forms.wiki import WikiEditModelForm

from django.views.decorators.csrf import csrf_exempt
from utils.encrypt import uid
from utils.tencent.cos import upload_file


def wiki(request, user_id):
    """wiki首页"""
    wiki_id = request.GET.get('wiki_id')

    # 如果传入的wiki_id不是数字
    if not wiki_id or not wiki_id.isdecimal():
        return render(request, "web/wiki.html")

    wiki_object = models.Wiki.objects.filter(id=wiki_id, user_id=request.tracer).first()

    print("wiki_object:", wiki_object)
    print("wiki_object.page_content:", wiki_object.page_content)

    return render(request, "web/wiki.html", {"wiki_object": wiki_object})


def wiki_add(request, user_id):
    """添加文章"""

    # 展示添加文章页面
    if request.method == "GET":
        form = WikiAddModelForm(request)
        return render(request, "web/wiki_add.html", {'form': form})

    # 文章校验
    form = WikiAddModelForm(request, data=request.POST)
    if form.is_valid():

        # 判断用户是否选择父文章,设置depth
        if form.instance.parent:
            form.instance.depth = form.instance.parent.depth + 1
        else:
            form.instance.depth = 1

        form.instance.user = request.tracer
        form.save()
        url = reverse('web:wiki', kwargs={'user_id': user_id})  # 跳转回wiki页面
        print("form is_valid")
        print("page", form.cleaned_data)
        print("url", url)
        return JsonResponse({"status": True, "data": url})
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({"status": False, 'error': form.errors})


def wiki_catalog(request, user_id):
    """wiki目录"""

    # 获取当前用户的所有wiki文章
    # data = models.Wiki.objects.filter(user=request.tracer).values("id", "page_title", "parent_id")
    data = models.Wiki.objects.filter(user=request.tracer).values("id", "page_title", "parent_id").order_by("depth",
                                                                                                            "id")
    # 将QuerySet转换为列表
    data_list = list(data)

    print("data_list:", data_list)

    return JsonResponse({"status": True, "data": data_list})


def wiki_delete(request, user_id, wiki_id):
    models.Wiki.objects.filter(id=wiki_id, user_id=user_id).delete()
    url = reverse('web:wiki', kwargs={'user_id': user_id})  # 跳转回wiki页面
    return redirect(url)


def wiki_edit(request, user_id, wiki_id):
    # 获取wiki对象
    wiki_object = models.Wiki.objects.filter(id=wiki_id, user_id=user_id).first()

    # wiki对象不存在
    if not wiki_object:
        url = reverse('web:wiki', kwargs={'user_id': user_id})  # 跳转回wiki页面
        return redirect(url)

    # 显示文章原始内容
    if request.method == "GET":
        form = WikiEditModelForm(request, instance=wiki_object)
        return render(request, "web/wiki_edit.html", {'form': form, 'wiki_object': wiki_object})

    # 提交修改
    if request.method == "POST":
        form = WikiEditModelForm(request, data=request.POST, instance=wiki_object)

        if form.is_valid():

            # 判断用户是否选择父文章,设置depth
            if form.instance.parent:
                form.instance.depth = form.instance.parent.depth + 1
            else:
                form.instance.depth = 1

            form.save()

            # 递归更新子节点的depth
            form.instance.update_children_depth()

            url = reverse('web:wiki', kwargs={'user_id': user_id})  # 跳转回wiki页面

            preview_url = "{0}?preview={1}".format(url, form.instance.id)  # 跳转到编辑后的文章页面

            return JsonResponse({"status": True, "data": preview_url})
        else:
            print("form not_valid")
            print(form.errors)
            return JsonResponse({"status": False, 'error': form.errors})


# 去掉csrf认证
@csrf_exempt
def wiki_upload(request, user_id):
    """markdown插件上传图片"""
    print("收到图片",request.FILES)

    # 图片对象
    image_object = request.FILES.get("editormd-image-file")

    if not image_object:
        # 将结果返回给markdown组件【失败】
        result = {
            "success": 0,
            "message": "文件不存在。",
        }
        return JsonResponse(result)

    # 检查文件扩展名
    filename = image_object.name        # 上传的文件名

    parts = filename.rsplit('.', 1)

    if len(parts) < 2:
        result = {
            "success": 0,
            "message": "文件缺少扩展名。",
            "url": None
        }
        return JsonResponse(result)

    ext = parts[1].lower()      # 文件后缀名

    # 随机文件名
    key = "{}.{}".format(uid(request.tracer.user_phone), ext)

    # 上传到COS
    try:
        image_url = upload_file(
            request.tracer.bucket,
            request.tracer.region,
            image_object,
            key
        )
    except Exception as e:
        # 可以记录日志，例如print或者使用logging模块
        print(f"上传到COS失败: {e}")
        result = {
            "success": 0,
            "message": "文件上传失败，请稍后再试。",
            "url": None
        }
        return JsonResponse(result)

    # 检查是否返回了有效的URL（假设上传成功返回非空URL）
    if not image_url:
        result = {
            "success": 0,
            "message": "文件上传失败，无法获取URL。",
            "url": None
        }
        return JsonResponse(result)

    # 成功返回
    result = {
        "success": 1,
        "message": None,
        "url": image_url,
    }
    return JsonResponse(result)
