# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import logging
from django.conf import settings


# 创建桶
def create_bucket(bucket, region="ap-guangzhou"):
    """
    创建桶
    :param bucket: 桶名称
    :param region: 区域
    :return:
    """
    secret_id = settings.COS_SECRET_ID

    secret_key = settings.COS_SECRET_KEY

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)

    client = CosS3Client(config)

    client.create_bucket(
        Bucket=bucket,
        ACL='public-read',
    )

    # 配置CORS规则
    cors_config = {
        'CORSRule': [
            {
                "AllowedOrigin": "*",
                "AllowedMethod": ["GET", "POST", "PUT", "DELETE", "HEAD"],
                "AllowedHeader": "*",
                "ExposeHeader": "*",
                "MaxAgeSeconds": 500
            }
        ]

    }

    # 设置CORS配置
    client.put_bucket_cors(
        Bucket=bucket,
        CORSConfiguration=cors_config
    )


# 上传文件到桶
def upload_file(bucket, region, file_object, key):
    # 把图片对象上传到当前用户的桶
    secret_id = settings.COS_SECRET_ID

    secret_key = settings.COS_SECRET_KEY

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)

    client = CosS3Client(config)

    response = client.upload_file_from_buffer(
        Bucket=bucket,
        Body=file_object,  # 文件对象
        Key=key,  # 文件名
    )

    return "https://{}.cos.{}.myqcloud.com/{}".format(bucket, region, key)


# 桶删除文件
def delete_file(bucket, region, key):
    secret_id = settings.COS_SECRET_ID

    secret_key = settings.COS_SECRET_KEY

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)

    client = CosS3Client(config)

    client.delete_object(
        Bucket=bucket,
        Key=key,  # 文件名
    )

# 桶批量删除文件
def delete_file_list(bucket, region, key_list):
    """

    :param bucket:
    :param region:

    :param key_list:传入的数据格式如下
    [
        {"Key": "file_name1"},
        {"Key": "file_name2"}
    ]

    :return:
    """
    secret_id = settings.COS_SECRET_ID

    secret_key = settings.COS_SECRET_KEY

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)

    client = CosS3Client(config)

    """
    批量删除文件
    objects = {
        "Quiet": "true",
        "Object": [
            {
                "Key": "file_name1"
            },
            {
                "Key": "file_name2"
            }
        ]
    }
    """

    objects = {
        "Quiet": "true",
        "Object": key_list
    }

    client.delete_objects(
        Bucket=bucket,
        Delete=objects,  # 文件列表
    )
