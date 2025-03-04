from django.urls import include, path, re_path

from web.views import account
from web.views import project
from web.views import wiki

app_name = 'web'

urlpatterns = [
    # 注册
    path('register/', account.register, name='register'),
    # path('verify/username/', account.verify_username, name='verify_username'),

    # 图片验证码
    path('img/code/', account.img_code, name='img_code'),

    # 手机号登录
    path('login/user_phone/', account.login_phone, name='login_phone'),

    # 用户名登录
    path('login/user_name/', account.login_name, name='login_name'),

    # 邮箱登录
    path('login/user_email/', account.login_email, name='login_email'),

    # 退出登录
    path('logout/', account.logout, name='logout'),

    # 首页
    path('index/', account.index, name='index'),

    # 帮助文档
    path('help/', account.help, name='help'),

    # 创建项目
    path('project/add/', project.project_add, name='project_add'),

    # 展示项目
    path('project/list/', project.project_list, name='project_list'),

    # wiki
    re_path(r'^(?P<user_id>\d+)/wiki/$', wiki.wiki, name='wiki'),

    # 新建文章
    re_path(r'^(?P<user_id>\d+)/wiki/add/$', wiki.wiki_add, name='wiki_add'),

    # 展示多级目录
    re_path(r'^(?P<user_id>\d+)/wiki/catalog/$', wiki.wiki_catalog, name='wiki_catalog'),

    # 星标项目
    re_path(r'^project/star/(?P<project_type>\w+)/(?P<project_id>\d+)/$', project.project_star, name='project_star'),

    # 取消星标
    re_path(r'^project/delete_star/(?P<project_type>\w+)/(?P<project_id>\d+)/$', project.project_delete_star, name='project_delete_star'),

    # 图像分割
    re_path(r'^project/image_segmentation/(?P<project_id>\d+)/$', project.project_image_segmentation, name='project_image_segmentation'),

    # 项目管理
    re_path(r'^project/manage/(?P<project_id>\d+)/$', project.project_manage, name='project_manage'),

]
