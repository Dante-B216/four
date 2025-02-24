from django.urls import include, path

from app01 import views

app_name = 'app01'

urlpatterns = [
    path('register/', views.register),
]