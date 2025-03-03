from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator

from web.forms.bootstrap import BootstrapForm

from web.forms.widgets import ColorRadioSelect

from web.models import Project


class ProjectModelForm(BootstrapForm, forms.ModelForm):

    bootstrap_class_exclude = ['color']

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
        fields = ['name','description']
        widgets = {
            'color': ColorRadioSelect(attrs={'class': 'color-radio'}),
        }

    def __init__(self, request, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request = request  # 把试图里面的对象传到Form中

    # 判断当前用户是否已创建过相同项目名称
    def clean_name(self):
        name = self.cleaned_data['name']
        exists = Project.objects.filter(name=name, user=self.request.tracer).exists()
        if exists:
            raise ValidationError("项目名称已存在。")

        return name
