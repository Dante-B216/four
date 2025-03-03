from django.utils.deprecation import MiddlewareMixin
from django.shortcuts import redirect
from web import models
from django.conf import settings

# 中间件
class AuthMiddleware(MiddlewareMixin):
    def process_request(self, request):
        """
        :param request:
        :return:
        如果用户已登录，则request中赋值
        """
        user_id = request.session.get('user_id', 0)

        print("user_id:", user_id)

        user_object = models.UserInfo.objects.filter(id=user_id).first()

        print("user_object:",user_object)

        request.tracer = user_object

        # 白名单：没有登录都可以访问的url
        # 获取当前用户正在访问的url
        # 检查url是否在白名单中，在则继续访问，不在则判断是否已经登录
        print(request.path_info)
        if request.path_info in settings.WHITE_REGEX_URL_LIST:
            return

        if not request.tracer:
            return redirect('/web/login/user_name/')

    def process_view(self, request, view, args, kwargs):
        # 如果url不是以'/web/project/manage/'开头，不做处理
        if not request.path_info.startswith('/web/project/manage/'):
            return

        # 获取当前项目id
        project_id = kwargs.get('project_id')
        print("project_id:", project_id)

        # 判断当前项目id是不是当前用户创建的项目
        project_object = models.Project.objects.filter(user=request.tracer, id=project_id).first()

        print("project_object:", project_object)

        request.project = project_object

        if project_object:
            return  # 是，继续
        return redirect('/web/project/list/')   # 不是，跳转回项目列表
