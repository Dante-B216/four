from django.utils.deprecation import MiddlewareMixin
from django.shortcuts import redirect
from web import models
from django.conf import settings


class AuthMiddleware(MiddlewareMixin):
    def process_request(self, request):
        """
        :param request:
        :return:
        如果用户已登录，则request中赋值
        """
        user_id = request.session.get('user_id', 0)

        print("user_id")
        print(user_id)

        user_object = models.UserInfo.objects.filter(id=user_id).first()

        print("user_object")
        print(user_object)

        request.tracer = user_object

        # 白名单：没有登录都可以访问的url
        # 获取当前用户正在访问的url
        # 检查url是否在白名单中，在则继续访问，不在则判断是否已经登录
        print(request.path_info)
        if request.path_info in settings.WHITE_REGEX_URL_LIST:
            return

        if not request.tracer:
            return redirect('/web/login/user_name/')
