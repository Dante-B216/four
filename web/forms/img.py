from web import models
from django import forms
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError

from utils import encrypt
from web.forms.bootstrap import BootstrapForm


class SegmentationResultChoicesModelForm(BootstrapForm, forms.ModelForm):
    class Meta:
        model = models.SegmentationResult
        fields = ["model_type"]