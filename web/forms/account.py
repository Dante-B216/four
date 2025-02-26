from django.core.exceptions import ValidationError

from web import models
from django import forms
from django.core.validators import RegexValidator
from utils import encrypt


class RegisterModelForm(forms.ModelForm):

    # 用户名验证
    user_name = forms.CharField(
        label='用户名',
        min_length=6,
        max_length=16,
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z0-9_-]{6,16}$',  # 正则表达式
                message='用户名只能包含字母、数字、下划线或短横线，长度为6-16位',
            )
        ]
    )

    # 邮箱验证
    user_email = forms.EmailField(
        label='邮箱',
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # 正则表达式
                message='邮箱格式错误'
            )
        ]
    )

    # 手机号验证
    user_phone = forms.CharField(
        label='手机号',
        validators=[
            RegexValidator(
                regex=r'^1[3456789]\d{9}$',  # 正则表达式
                message='手机号格式错误'
            )
        ]
    )

    # 密码隐藏
    user_pw = forms.CharField(
        label='密码',
        min_length=8,
        max_length=32,
        error_messages={
            'min_length': "密码长度不能小于8个字符",
            'max_length': "密码长度不能大于32个字符"
        },
        widget=forms.PasswordInput
    )

    # 二次输入密码
    confirm_user_pw = forms.CharField(
        label='重复密码',
        min_length=8,
        max_length=32,
        error_messages={
            'min_length': "重复密码长度不能小于8个字符",
            'max_length': "重复密码长度不能大于32个字符"
        },
        widget=forms.PasswordInput
    )

    class Meta:
        model = models.UserInfo
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            field.widget.attrs['placeholder'] = '请输入%s' % (field.label)

    def clean_user_name(self):
        user_name = self.cleaned_data['user_name']

        # 检查用户名是否已存在
        exists = models.UserInfo.objects.filter(user_name=user_name).exists()
        if exists:
            raise ValidationError('用户名已注册')

        return user_name

    def clean_user_email(self):
        user_email = self.cleaned_data['user_email']

        exists = models.UserInfo.objects.filter(user_email=user_email).exists()
        if exists:
            raise ValidationError('邮箱已注册')

        return user_email

    def clean_user_phone(self):
        user_phone = self.cleaned_data['user_phone']

        exists = models.UserInfo.objects.filter(user_phone=user_phone).exists()

        if exists:
            raise ValidationError('手机号已注册')

        return user_phone

    def clean_user_pw(self):
        user_pw = self.cleaned_data['user_pw']

        # 加密并返回
        return encrypt.md5(user_pw)

    def clean_confirm_user_pw(self):
        user_pw = self.cleaned_data['user_pw']

        confirm_user_pw = encrypt.md5(self.cleaned_data['confirm_user_pw'])

        if confirm_user_pw != user_pw:
            raise ValidationError('两次密码不一致')

        return confirm_user_pw



# class verifyUsernameForm(forms.Form):
#     user_name = forms.CharField(
#         label='用户名',
#         min_length=6,
#         max_length=16,
#         validators=[
#             RegexValidator(
#                 regex=r'^[a-zA-Z0-9_-]{6,16}$',  # 修正后的正则表达式
#                 message='用户名只能包含字母、数字、下划线或短横线，长度为6-16位',
#                 code='invalid_username'
#             )
#         ]
#     )
#
#     def __init__(self,request,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#         self.request = request  # 把试图里面的对象传到Form中
#
#     def clean_user_name(self):
#         # 手机号校验钩子
#         user_name=self.cleaned_data['user_name']
#
#         # 检查用户名是否已存在
#         query = models.UserInfo.objects.filter(user_name__iexact=user_name)
#
#         if query.exists():
#             raise ValidationError('该用户名已被注册，请换一个试试', code='username_exists')
#
#         return user_name
#
#
#
#
#
#     pass
