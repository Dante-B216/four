from django.utils.deprecation import MiddlewareMixin
from web import models


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
