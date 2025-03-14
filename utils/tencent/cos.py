# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import logging
from django.conf import settings

import json
import os

from sts.sts import Sts, CIScope, Scope
from qcloud_cos import CosServiceError


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

    data = client.delete_object(
        Bucket=bucket,
        Key=key,  # 文件名
    )

    return data


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


# 获取临时凭证
def get_credential(bucket, region, ):
    config = {
        # 临时密钥有效时长，单位是秒，1800s=30min
        'duration_seconds': 1800,

        # 固定密钥
        'secret_id': settings.COS_SECRET_ID,
        'secret_key': settings.COS_SECRET_KEY,

        # 换成你的 bucket
        'bucket': bucket,

        # 换成 bucket 所在地区
        'region': region,

        # 这里改成允许的路径前缀，可以根据自己网站的用户登录态判断允许上传的具体路径
        # 例子： a.jpg 或者 a/* 或者 * (使用通配符*存在重大安全风险, 请谨慎评估使用)
        'allow_prefix': ['*'],

        # 密钥的权限列表。简单上传和分片需要以下的权限，其他权限列表请看 https://cloud.tencent.com/document/product/436/31923
        'allow_actions': [
            # 简单上传
            # 'name/cos:PostObject',

            # 'name/cos:PutObject',
            # 分片上传
            # 'name/cos:InitiateMultipartUpload',
            # 'name/cos:ListMultipartUploads',
            # 'name/cos:ListParts',
            # 'name/cos:UploadPart',
            # 'name/cos:CompleteMultipartUpload',
            "*"
        ],

    }

    try:
        sts = Sts(config)
        response = sts.get_credential()  # 临时凭证
        print('get data : ' + json.dumps(dict(response), indent=4))
        return response  # 返回临时凭证
    except Exception as e:
        print(e)


def download_file(bucket, region, key):
    # 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    secret_id = settings.COS_SECRET_ID

    secret_key = settings.COS_SECRET_KEY

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)

    client = CosS3Client(config)

    # 使用高级接口下载一次，不重试，此时没有使用断点续传的功能
    response = client.download_file(
        Bucket=bucket,
        Key=key,
        DestFilePath="web/views/handle_img/" + key
    )


# 删除桶
def delete_bucket(bucket, region):
    # 删除所有文件
    # 删除所有碎片

    secret_id = settings.COS_SECRET_ID

    secret_key = settings.COS_SECRET_KEY

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)

    client = CosS3Client(config)

    try:
        # 找到所有文件并删除
        while True:
            part_object = client.list_objects(bucket)

            contents = part_object.get('Contents')

            if not contents:
                break

            objects = {
                "Quiet": "true",
                "Object": [{"Key": item["Key"]} for item in contents]
            }

            # 批量删除
            client.delete_objects(bucket, objects)

            if part_object["IsTruncated"] == "false":
                break

        # 找到所有碎片并删除
        while True:
            part_uploads = client.list_multipart_uploads(bucket)
            uploads = part_uploads.get("Upload")
            if not uploads:
                break

            for item in uploads:
                client.abort_multipart_upload(bucket, item["Key"], item["UploadId"])

            if part_object["IsTruncated"] == "false":
                break

        # 删除桶
        client.delete_bucket(bucket)

    except CosServiceError as e:
        print(e)
