# 创建一个Library实例，用于注册自定义模板标签
from django import template

# 导入Django的reverse函数，用于生成URL
from django.urls import reverse

# 创建一个Library实例，用于注册自定义模板标签
register = template.Library()

# 使用inclusion_tag装饰器注册一个包含标签
# 该标签会渲染'web/inclusion/basic_menu_list.html'模板
@register.inclusion_tag('web/inclusion/basic_menu_list.html')
def basic_menu_list(request):
    # 定义一个菜单数据列表，每个菜单项包含标题和URL
    data_list = [
        {'title': '首页', 'url': reverse('web:index')},
        {'title': '帮助文档', 'url': reverse('web:help')}
    ]

    # 遍历菜单数据列表，为每个菜单项添加class属性
    for item in data_list:
        item['class'] = "nav-link"      # 默认class为"nav-link"

        if request.path_info.startswith(item['url']):       # 检查当前请求的路径是否以菜单项的URL开头
            print("request.path_info", request.path_info)
            print("item['url']", item['url'])
            item['class'] = "nav-link active"       # 将class设置为"nav-link active"，表示当前菜单项是激活状态

    return {'data_list': data_list}     # 返回一个包含菜单数据的字典，供模板渲染使用
