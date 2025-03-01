from django import forms
from django.core.validators import RegexValidator

from web.forms.bootstrap import BootstrapForm
from web.models import Project


class ProjectModelForm(BootstrapForm, forms.ModelForm):
    name = forms.CharField(
        label='项目名称',
        min_length=2,
        max_length=32,
        validators=[
            RegexValidator(
                regex=r'^.{2,32}$',  # 正则表达式
                message='项目名称长度为2-32位。'
            )
        ]
    )

    class Meta:
        model = Project
        fields = ['name', 'description']
