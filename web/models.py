from django.db import models

class UserInfo(models.Model):
    user_name = models.CharField(verbose_name='用户名', max_length=32)
    user_email = models.EmailField(verbose_name='邮箱',max_length=32)     # 内置正则表达式
    user_phone = models.CharField(verbose_name='手机号',max_length=32)
    user_pw = models.CharField(verbose_name='密码', max_length=32)
