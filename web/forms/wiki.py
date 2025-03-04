from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator

from web import models
from web.forms.bootstrap import BootstrapForm


class WikiAddModelForm(BootstrapForm, forms.ModelForm):
    page_title = forms.CharField(
        label='文章标题',
        min_length=2,
        max_length=255,
        validators=[
            RegexValidator(
                regex=r'^.{2,32}$',  # 正则表达式
                message='文章标题长度为2-255位。'
            )
        ]
    )

    class Meta:
        model = models.Wiki
        fields = ['page_title', 'page_content', 'parent']

    def __init__(self, request, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request = request  # 把试图里面的对象传到Form中

        total_data_list = [("", "请选择")]
        data_list = models.Wiki.objects.filter(user=request.tracer).values_list('id', 'page_title')
        total_data_list.extend(data_list)

        self.fields['parent'].choices = total_data_list

    def clean_page_title(self):
        page_title = self.cleaned_data['page_title']

        exits = models.Wiki.objects.filter(page_title=page_title,user=self.request.tracer).exists()

        if exits:
            raise ValidationError("文章标题已存在。")

        return page_title


class WikiEditModelForm(BootstrapForm, forms.ModelForm):
    page_title = forms.CharField(
        label='文章标题',
        min_length=2,
        max_length=255,
        validators=[
            RegexValidator(
                regex=r'^.{2,32}$',  # 正则表达式
                message='文章标题长度为2-255位。'
            )
        ]
    )

    class Meta:
        model = models.Wiki
        fields = ['page_title', 'page_content', 'parent']

    def __init__(self, request, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request = request  # 把试图里面的对象传到Form中

        total_data_list = [("", "请选择")]
        data_list = models.Wiki.objects.filter(user=request.tracer).values_list('id', 'page_title')
        total_data_list.extend(data_list)

        self.fields['parent'].choices = total_data_list

    def clean_page_title(self):
        page_title = self.cleaned_data['page_title']

        # 获取当前用户的所有同标题文章
        queryset = models.Wiki.objects.filter(
            page_title=page_title,
            user=self.request.tracer
        )

        # 编辑时排除自己（新建时instance.id不存在）
        if self.instance.pk:
            queryset = queryset.exclude(id=self.instance.pk)

        if queryset.exists():
            raise ValidationError("文章标题已存在。")

        return page_title