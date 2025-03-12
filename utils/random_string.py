# 理论上不会重复的随机字符串
import uuid

def generate_random_string():

    uid = str(uuid.uuid4())

    return uid
