import argparse  # 用于解析命令行参数的模块
import logging  # 用于记录日志的模块
import os  # 提供与操作系统交互的功能，例如文件路径管理
import sys

import numpy as np  # 用于科学计算的库，特别是数组和矩阵操作
import torch  # PyTorch深度学习框架，支持张量运算和自动求导
import torch.nn.functional as F  # PyTorch中包含常用神经网络功能的模块，例如激活函数和损失函数
from PIL import Image  # Python图像处理库，用于加载和处理图像

from ml.tools.data_loading import BasicDataset  # 从自定义模块中导入BasicDataset类，用于数据加载和预处理
from ml.netModelsTools import UNet  # 从自定义模块中导入UNet模型
from ml.netModelsTools import UNetPlusPlus  # 从自定义模块中导入UNet++模型
from ml.netModelsTools import U2Net  # 从自定义模块中导入U2Net模型
from ml.tools.utils import plot_img_and_mask  # 从自定义模块中导入函数，用于可视化图像及其分割掩膜
import cv2  # 导入OpenCV库用于轮廓检测


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    """使用训练好的模型对输入图像进行分割预测。

        参数：
            - net: 神经网络模型，通常是预训练好的分割模型（如UNet）。
            - full_img: 输入的完整图像，PIL.Image格式。
            - device: 设备类型（如"cpu"或"cuda"）。
            - scale_factor: 图像缩放因子，用于调整图像大小以适应模型输入尺寸。
            - out_threshold: 输出阈值，用于将连续值的预测转换为二值掩膜。

        返回：
            - mask: 预测的分割掩膜，numpy数组格式。
     """

    net.eval()  # 设置模型为评估模式，禁用dropout等训练专用操作

    # 预处理输入图像：调整大小、归一化，并转换为PyTorch张量
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)  # 添加一个批次维度，形状变为 (1, C, H, W)
    img = img.to(device=device, dtype=torch.float32)  # 将张量移动到指定设备，并转换为浮点类型

    # 禁用梯度计算，以减少内存占用和加速推理
    with torch.no_grad():
        output = net(img).cpu()  # 前向传播，得到模型输出，并将结果移回CPU

        # 使用双线性插值将输出调整到与原始图像相同的大小
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        if net.n_classes > 1:  # 多分类分割
            mask = output.argmax(dim=1)  # 找到每个像素所属的类别
        else:  # 二分类分割
            mask = torch.sigmoid(output) > out_threshold  # 应用sigmoid并根据阈值生成二值掩膜

    return mask[0].long().squeeze().numpy()  # 返回预测的掩膜，并转换为numpy数组


def mask_to_image(mask: np.ndarray, mask_values):
    """将预测掩膜转换为可保存的图像格式。

        参数：
            - mask: numpy数组格式的分割掩膜。
            - mask_values: 掩膜值，用于指定每个类别的像素值。

        返回：
            - 转换后的图像，PIL.Image格式。
    """

    # 如果mask_values是嵌套列表，初始化三维输出数组（多通道）
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:  # 如果mask_values是布尔值范围，初始化二值掩膜
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:  # 其他情况，初始化单通道的8位无符号整数数组
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # 如果掩膜是三维的，选择每个像素值最大的通道索引作为类别
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # 遍历每个类别索引和对应的像素值，将掩膜中该类别的像素赋值为指定值
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    # 将numpy数组转换为PIL.Image并返回
    return Image.fromarray(out)


# def overlay_mask_on_image(original_img, mask, color=(255, 0, 0), alpha=0.3):
#     """
#     将分割掩码以指定颜色和透明度叠加到原始图像上。
#
#     参数:
#         original_img (PIL.Image): 原始图像。
#         mask (numpy.ndarray): 分割掩码，数值为0和1。
#         color (tuple): 覆盖颜色，RGB格式，默认为红色(255, 0, 0)。
#         alpha (float): 覆盖颜色的透明度，0.0为完全透明，1.0为不透明。
#
#     返回:
#         PIL.Image: 叠加后的图像。
#     """
#     original = original_img.convert('RGB')
#     overlay = Image.new('RGB', original.size, color)
#     mask_bool = mask.astype(bool)
#
#     original_np = np.array(original)
#     overlay_np = np.array(overlay)
#
#     blended_np = np.where(mask_bool[:, :, np.newaxis],
#                           (original_np * (1 - alpha) + overlay_np * alpha).astype(np.uint8),
#                           original_np)
#     blended_img = Image.fromarray(blended_np)
#     return blended_img

def overlay_mask_on_image(original_img, mask, color=(255, 0, 0), alpha=0.3,
                          contour_color=(63, 161, 95), contour_width=1):
    """
    将分割掩码以半透明颜色叠加到原始图像，并用实线标出分割区域最外围轮廓

    参数:
        original_img (PIL.Image): 原始图像
        mask (numpy.ndarray): 二值分割掩码（0和1）
        color (tuple): 覆盖颜色，RGB格式，默认红色
        alpha (float): 颜色透明度（0-1）
        contour_color (tuple): 轮廓线颜色，BGR格式，默认绿色
        contour_width (int): 轮廓线宽度

    返回:
        PIL.Image: 带轮廓线的叠加图像
    """
    # 将原始图像转换为RGB格式的numpy数组
    original_np = np.array(original_img.convert('RGB'))

    # 创建颜色叠加层
    overlay = Image.new('RGB', original_img.size, color)
    overlay_np = np.array(overlay)

    # 混合原始图像和颜色叠加层
    mask_3d = mask[:, :, np.newaxis]  # 增加通道维度
    blended_np = np.where(mask_3d,
                          (original_np * (1 - alpha) + overlay_np * alpha).astype(np.uint8),
                          original_np)

    # 转换为OpenCV格式（BGR）
    blended_cv = cv2.cvtColor(blended_np, cv2.COLOR_RGB2BGR)

    # 准备轮廓检测用的掩码
    mask_uint8 = mask.astype(np.uint8) * 255

    # 查找轮廓（只检测外部轮廓）
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,  # 只检测最外层轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓存储
    )

    # 在图像上绘制所有检测到的轮廓
    if len(contours) > 0:
        cv2.drawContours(
            blended_cv,
            contours,
            -1,  # 绘制所有轮廓
            contour_color,
            contour_width,
            lineType=cv2.LINE_AA  # 抗锯齿线型
        )

    # 转换回RGB格式
    result_rgb = cv2.cvtColor(blended_cv, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def get_output_filename(file_path):
    """生成输出文件名列表。

        如果未提供输出文件名，则基于输入文件名生成默认的输出文件名。
        默认的输出文件名格式为：<输入文件名>_OUT.png

        参数：
            - args: 包含输入和输出文件名信息的命名空间对象。

        返回：
            - 输出文件名列表。
    """
    # 获取文件所在目录的路径
    dir_path = os.path.dirname(file_path)

    # 获取文件名和扩展名
    base_name = os.path.basename(file_path)

    file_name, ext = os.path.splitext(base_name)

    # 构造新文件名（在原文件名后添加"_OUT"）

    new_file_name = f"{file_name}_OUT{ext}"

    # 返回新文件名
    return new_file_name


def get_overlay_filename(file_path):
    """生成输出文件名列表。

        如果未提供输出文件名，则基于输入文件名生成默认的输出文件名。
        默认的输出文件名格式为：<输入文件名>_OUT.png

        参数：
            - args: 包含输入和输出文件名信息的命名空间对象。

        返回：
            - 输出文件名列表。
    """
    # 获取文件所在目录的路径
    dir_path = os.path.dirname(file_path)

    # 获取文件名和扩展名
    base_name = os.path.basename(file_path)

    file_name, ext = os.path.splitext(base_name)

    new_file_name = f"{file_name}_overlay{ext}"

    # 返回新文件名
    return new_file_name


def segment_image(img_path, model_path):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')  # 配置日志输出格式

    in_file = img_path  # 输入文件

    print('in_file', in_file)

    out_file = get_output_filename(in_file)

    print('out_file', out_file)

    # 设置设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    # 加载模型参数
    logging.info(f'Loading model from {model_path}')

    state_dict = torch.load(model_path, map_location=device)
    model_name = state_dict.pop('model_name', None)  # 从模型参数中提取模型名称
    mask_values = state_dict.pop('mask_values', [0, 1])  # 从模型参数中提取掩膜值

    # 根据模型名称实例化对应的模型
    if model_name == 'unet++':
        net = UNetPlusPlus(n_channels=1, n_classes=2, bilinear=False)
    elif model_name == 'u2net':
        net = U2Net(n_channels=1, n_classes=2)
    elif model_name == 'unet_cs':
        net = UNet(n_channels=1, n_classes=2, bilinear=False,
                   c_attention=True, s_attention=True)
    elif model_name == 'unet_c':
        net = UNet(n_channels=1, n_classes=2, bilinear=False,
                   c_attention=True, s_attention=False)
    elif model_name == 'unet_s':
        net = UNet(n_channels=1, n_classes=2, bilinear=False,
                   c_attention=False, s_attention=True)
    elif model_name == 'unet':
        net = UNet(n_channels=1, n_classes=2, bilinear=False,
                   c_attention=False, s_attention=False)
    else:
        raise ValueError(f'Model {model_name} not recognized')  # 如果模型名称无效，抛出异常

    net.to(device=device)  # 将模型移动到指定设备

    net.load_state_dict(state_dict)  # 加载模型参数

    logging.info(f'Model {model_name} loaded!')

    logging.info(f'Predicting image {in_file} ...')  # 输出当前处理的文件名

    current_img = Image.open(in_file)  # 加载图像

    # 使用模型进行分割预测
    mask = predict_img(net=net, full_img=current_img, device=device)

    print("Unique mask values:", np.unique(mask))  # Unique mask values: [0 1]

    print(f"mask数组数据类型: {mask.dtype}")  # mask数组数据类型: int64
    print(f"mask数组数值范围: {mask.min()}~{mask.max()}")  # mask数组数值范围: 0~1

    # 保存结果掩码文件
    # out_filename = out_file  # 获取当前输出文件名
    #
    # result = mask_to_image(mask, mask_values)  # 将预测的掩码转换为图像格式
    #
    # result.save(out_filename)  # 保存转换后的图像到指定路径
    #
    # logging.info(f'Mask saved to {out_filename}')  # 记录日志，提示掩码已保存

    # 生成并保存叠加图像
    overlay = overlay_mask_on_image(current_img, mask)
    overlay_filename = "web/views/handle_img/" + get_overlay_filename(in_file)
    overlay.save(overlay_filename)
    logging.info(f'Overlay image saved to {overlay_filename}')

    return overlay_filename

    # 可视化图像和对应的掩码
    # logging.info(f'Visualizing results for image {in_file}, close to continue...')  # 提示用户当前图像正在可视化
    #
    # plot_img_and_mask(current_img, mask)  # 显示原始图像及其预测掩码


# if __name__ == '__main__':
#     img = "D:/xuniCpan/Graduation Design/graduationDesign/ml/test_img/P17-0080.png"
#     model_path = "D:/xuniCpan/Graduation Design/graduationDesign/ml/netModels/unet_s.pth"
#
#     segment_image(img_path=img, model_path=model_path)
