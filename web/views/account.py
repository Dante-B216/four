from django.contrib.auth.hashers import check_password
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect

import web
from web import models
from web.forms.account import RegisterModelForm
from web.forms.account import LoginPhoneModelForm
from web.forms.account import LoginNameModelForm
from web.forms.account import LoginEmailModelForm


def register(request):
    if request.method == 'GET':
        form = RegisterModelForm()
        return render(request, 'web/register.html', {'form': form})

    form = RegisterModelForm(request.POST)
    if form.is_valid():
        print("form is_valid")
        print(form.cleaned_data)
        # 验证通过，写入数据库（密码加密）
        instance = form.save()
        return JsonResponse({'status': True, 'data': '/web/login/user_name/'})
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({'status': False, 'error': form.errors})


# def verify_username(request):
#     form = verifyUsernameForm(request, data=request.GET)
#
#     # 校验用户名不能为空，格式正确
#     if form.is_valid():
#         return JsonResponse({'status': 'true'})
#
#     return JsonResponse({'status': 'false', 'error': form.errors})

# 手机号登录
def login_phone(request):

    if request.method == 'GET':
        form = LoginPhoneModelForm(request)
        return render(request, 'web/login_phone.html', {'form': form})

    # request.method == 'POST':
    # 获取表单数据
    form = LoginPhoneModelForm(request, data=request.POST)
    # 表单验证通过
    if form.is_valid():
        print("form is_valid")
        print(form.cleaned_data)

        user_phone = form.cleaned_data['user_phone']
        user_pw = form.cleaned_data['user_pw']

        # 数据库中找到user_phone对应的用户
        user = models.UserInfo.objects.get(user_phone=user_phone)

        print(user.user_pw)
        print(user_pw)

        # 如果输入的密码和数据库用户密码一样
        if user.user_pw == user_pw:
            # 存储用户ID到session
            request.session['user_id'] = user.id

            return JsonResponse({'status': True, 'data': '/web/index'})     # 跳转首页
        else:
            return JsonResponse({'status': False, 'error': {'user_pw': ['密码错误。']}})     # 在user_pw下面报错：密码错误。
    # 表单验证不通过
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({'status': False, 'error': form.errors})    # 把错误显示在表单上


# 生成图片验证码
def img_code(request):
    from utils.img_code import check_code
    from io import BytesIO

    # 生成图片验证码
    image_object, code = check_code()

    # 把验证码写入到session
    request.session['image_object'] = code
    # 60s后失效
    request.session.set_expiry(60)

    # 保存图片到内存
    stream = BytesIO()
    image_object.save(stream, 'png')

    # 将内存中的图片返回给网页
    return HttpResponse(stream.getvalue())

def login_name(request):

    if request.method == 'GET':
        form = LoginNameModelForm(request)
        return render(request, 'web/login_name.html', {'form': form})

    # request.method == 'POST':
    # 获取表单数据
    form = LoginNameModelForm(request, data=request.POST)
    # 表单验证通过
    if form.is_valid():
        print("form is_valid")
        print(form.cleaned_data)

        user_name = form.cleaned_data['user_name']
        user_pw = form.cleaned_data['user_pw']

        # 数据库中找到user_phone对应的用户
        user = models.UserInfo.objects.get(user_name=user_name)

        print(user.user_pw)
        print(user_pw)

        # 如果输入的密码和数据库用户密码一样
        if user.user_pw == user_pw:
            # 存储用户ID到session
            request.session['user_id'] = user.id

            return JsonResponse({'status': True, 'data': '/web/index'})     # 跳转首页
        else:
            return JsonResponse({'status': False, 'error': {'user_pw': ['密码错误。']}})     # 在user_pw下面报错：密码错误。
    # 表单验证不通过
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({'status': False, 'error': form.errors})    # 把错误显示在表单上


def login_email(request):
    if request.method == 'GET':
        form = LoginEmailModelForm(request)
        return render(request, 'web/login_email.html', {'form': form})

    # request.method == 'POST':
    # 获取表单数据
    form = LoginEmailModelForm(request, data=request.POST)
    # 表单验证通过
    if form.is_valid():
        print("form is_valid")
        print(form.cleaned_data)

        user_email = form.cleaned_data['user_email']
        user_pw = form.cleaned_data['user_pw']

        # 数据库中找到user_phone对应的用户
        user = models.UserInfo.objects.get(user_email=user_email)

        print(user.user_pw)
        print(user_pw)

        # 如果输入的密码和数据库用户密码一样
        if user.user_pw == user_pw:
            # 存储用户ID到session
            request.session['user_id'] = user.id

            return JsonResponse({'status': True, 'data': '/web/index'})  # 跳转首页
        else:
            return JsonResponse({'status': False, 'error': {'user_pw': ['密码错误。']}})  # 在user_pw下面报错：密码错误。
    # 表单验证不通过
    else:
        print("form not_valid")
        print(form.errors)
        return JsonResponse({'status': False, 'error': form.errors})  # 把错误显示在表单上

# 退出
def logout(request):
    request.session.flush()
    return redirect('/web/index')

def index(request):
    return render(request, 'web/index.html')
