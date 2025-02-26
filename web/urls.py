from django.urls import include, path

from web.views import account

app_name = 'web'

urlpatterns = [
    path('register/', account.register, name='register'),
    # path('verify/username/', account.verify_username, name='verify_username'),
    path('login/user_phone/' , account.login_phone , name='login_phone'),

]
