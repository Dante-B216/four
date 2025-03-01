# 导入所需的Python标准库模块
import os  # 用于处理文件和目录路径
import sys  # 用于处理Python解释器相关的操作，如添加模块搜索路径
import django  # 导入Django框架

# 获取当前脚本文件的绝对路径，并找到项目根目录的路径
# os.path.abspath(__file__) 获取当前脚本的绝对路径
# os.path.dirname() 获取该路径的上一级目录
# 这里通过两次 os.path.dirname() 获取项目根目录的路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录添加到Python的模块搜索路径中
# 这样Python解释器可以找到项目中的模块
sys.path.append(base_dir)

# 设置Django的环境变量，指定Django的配置文件（settings.py）所在的模块
# "graduationDesign.settings" 是Django项目的配置文件路径
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "graduationDesign.settings")

# 初始化Django环境，加载Django的设置和应用程序
# 这一步是必须的，因为在脚本中使用Django的ORM（对象关系映射）之前，需要先初始化Django
django.setup()

# 导入项目中的模型模块和工具模块
from web import models  # 导入web应用中的models模块，用于操作数据库
from utils import encrypt  # 导入utils模块中的encrypt工具，用于加密密码

# 使用Django的ORM向数据库中的UserInfo表插入一条数据
# models.UserInfo.objects.create() 是Django ORM提供的创建新记录的方法
models.UserInfo.objects.create(
    user_name="qwerty",
    user_email="qwerty@qq.com",
    user_phone="13652955599",
    user_pw=encrypt.md5("qwe12345678")
)
