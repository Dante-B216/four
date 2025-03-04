from django import forms
from web import models
from web.forms.bootstrap import BootstrapForm


class WikiModelForm(BootstrapForm, forms.ModelForm):
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
