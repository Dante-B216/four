from django.urls import include, path

from web.views import account

app_name = 'web'

urlpatterns = [
    path('register/', account.register, name='register'),
]
