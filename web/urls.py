from django.urls import include, path

from web.views import account

app_name = 'web'

urlpatterns = [
    path('register/', account.register, name='register'),
    # path('verify/username/', account.verify_username, name='verify_username'),
    path('img/code/', account.img_code, name='img_code'),

    path('login/user_phone/', account.login_phone, name='login_phone'),
    path('login/user_name/', account.login_name, name='login_name'),
    path('login/user_email/', account.login_email, name='login_email'),
]
