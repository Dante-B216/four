from collections import deque

from django.db import models


class UserInfo(models.Model):
    user_name = models.CharField(verbose_name='用户名', max_length=32, db_index=True)  # db_index=True创建索引

    user_email = models.EmailField(verbose_name='邮箱', max_length=32)  # 内置正则表达式

    user_phone = models.CharField(verbose_name='手机号', max_length=32)

    user_pw = models.CharField(verbose_name='密码', max_length=32)


class Project(models.Model):
    COLOR_CHOICES = (
        (1, "#56b8eb"),
        (2, "#f28033"),
        (3, "#ebc656"),
        (4, "#a2d148"),
        (5, "#20bfa4"),
        (6, "#7461c2"),
        (7, "#20bfa3"),
    )

    name = models.CharField(verbose_name='项目名称', max_length=32)

    description = models.TextField(verbose_name='项目描述', max_length=300)

    color = models.SmallIntegerField(verbose_name='项目颜色', choices=COLOR_CHOICES, default=1)

    star = models.BooleanField(verbose_name='项目星标', default=False)

    project_time = models.DateTimeField(verbose_name='创建时间', auto_now_add=True)

    user = models.ForeignKey(UserInfo, on_delete=models.CASCADE)  # 外键


class OriginalImage(models.Model):
    original_img = models.ImageField(verbose_name='本地上传图像')

    original_img_time = models.DateTimeField(verbose_name='上传时间', auto_now_add=True)

    project = models.ForeignKey(Project, on_delete=models.CASCADE)  # 外键


class SegmentationResult(models.Model):
    MODEL_CHOICES = (
        (1, "unet"),
        (2, "unet_c"),
        (3, "unet_s"),
        (4, "unet_cs"),
        (5, "unet++"),
        (6, "u2net"),
    )

    model_type = models.SmallIntegerField(verbose_name='分割模型', choices=MODEL_CHOICES)

    result_img = models.ImageField(verbose_name='分割结果图像')

    result_img_time = models.DateTimeField(verbose_name='分割完成时间', auto_now_add=True)

    original_img = models.ForeignKey(OriginalImage, on_delete=models.CASCADE)  # 外键


class Wiki(models.Model):
    page_title = models.CharField(verbose_name='文章标题', max_length=255)

    page_content = models.TextField(verbose_name='文章内容')

    # auto_now_add=True：当Wiki实例第一次创建时，page_time字段会自动设置为当前时间
    page_time = models.DateTimeField(verbose_name='创建时间', auto_now_add=True)  # 仅在创建时设置

    # auto_now=True：每次保存模型实例时都会更新为当前时间
    update_time = models.DateTimeField(verbose_name='更新时间', auto_now=True)  # 每次修改时更新

    user = models.ForeignKey(UserInfo, on_delete=models.CASCADE)  # 外键

    # 自关联
    # null=True, blank=True：这些参数允许parent字段为空
    parent = models.ForeignKey('Wiki', on_delete=models.CASCADE, null=True, blank=True, related_name='children',
                               verbose_name='父文章')

    # 深度
    depth = models.IntegerField(verbose_name='深度', default=1)

    def __str__(self):
        return self.page_title

    # 父节点depth修改后，修改所有子节点的depth
    def update_children_depth(self):
        queue = deque([self])
        while queue:
            current_node = queue.popleft()
            children = current_node.children.all()
            for child in children:
                child.depth = current_node.depth + 1
                child.save(update_fields=['depth'])
                queue.append(child)
