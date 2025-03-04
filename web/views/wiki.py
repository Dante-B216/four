from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from web import models
from django.urls import reverse

from web.forms.wiki import WikiModelForm


def wiki(request, user_id):
    """wiki首页"""
    wiki_id = request.GET.get('wiki_id')

    # 如果传入的wiki_id不是数字
    if not wiki_id or not wiki_id.isdecimal():
        return render(request, "web/wiki.html")

    wiki_object = models.Wiki.objects.filter(id=wiki_id, user_id=request.tracer).first()

    print("wiki_object:", wiki_object)

    return render(request, "web/wiki.html", {"wiki_object": wiki_object})


def wiki_add(request, user_id):
    """添加文章"""

    # 展示添加文章页面
    if request.method == "GET":
        form = WikiModelForm(request)
        return render(request, "web/wiki_add.html", {'form': form})

    # 文章校验
    form = WikiModelForm(request, data=request.POST)
    if form.is_valid():

        # 判断用户是否选择父文章,设置depth
        if form.instance.parent:
            form.instance.depth = form.instance.parent.depth + 1
        else:
            form.instance.depth = 1

        form.instance.user = request.tracer
        form.save()
        url = reverse('web:wiki', kwargs={'user_id': user_id})      # 跳转回wiki页面
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
