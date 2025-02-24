from django.shortcuts import render

# Create your views here.

from django import forms
from app01 import models

from django.core.validators import RegexValidator

from io import BytesIO
from django.http import HttpResponse
from django.views import View
from PIL import Image, ImageDraw, ImageFont
import random

class RegisterModelForm(forms.ModelForm):
    # 用户名
    user_name = forms.CharField(label='用户名')

    # 邮箱
    user_email = forms.EmailField(label='邮箱')

    # 手机号验证
    user_phone = forms.CharField(label='手机号', validators=[RegexValidator(r'^1[3-9]\d{9}$', '手机号格式错误')])

    # 密码隐藏
    user_pw = forms.CharField(label='密码', widget=forms.PasswordInput)

    # 二次输入密码
    confirm_user_pw = forms.CharField(label='重复密码', widget=forms.PasswordInput)


    class Meta:
        model = models.UserInfo
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            field.widget.attrs['placeholder'] = '请输入%s' % (field.label)


def register(request):
    form = RegisterModelForm()
    return render(request, "app01/register.html", {'form': form})
