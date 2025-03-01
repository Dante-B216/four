from django.urls import include, path

from web.views import account
from web.views import project

app_name = 'web'

urlpatterns = [
    path('register/', account.register, name='register'),
    # path('verify/username/', account.verify_username, name='verify_username'),
    path('img/code/', account.img_code, name='img_code'),

    path('login/user_phone/', account.login_phone, name='login_phone'),
    path('login/user_name/', account.login_name, name='login_name'),
    path('login/user_email/', account.login_email, name='login_email'),

    path('logout/', account.logout, name='logout'),
    path('index/', account.index, name='index'),

    path('project/add/', project.project_add, name='project_add'),
    path('project/list/', project.project_list, name='project_list'),
]
