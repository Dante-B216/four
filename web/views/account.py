from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from web.forms.account import RegisterModelForm


def register(request):
    if request.method == 'GET':
        form = RegisterModelForm()
        return render(request, 'web/register.html', {'form': form})

    form = RegisterModelForm(request.POST)

    if form.is_valid():
        print(form.cleaned_data)
        # 验证通过，写入数据库（密码加密）
        instance = form.save()
        return JsonResponse({'status': True, 'data': '/web/login'})
    else:
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
