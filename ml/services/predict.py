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


def predict_img(net,full_img,device,scale_factor=1,out_threshold=0.5):
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

    net.eval()    # 设置模型为评估模式，禁用dropout等训练专用操作

    # 预处理输入图像：调整大小、归一化，并转换为PyTorch张量
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)    # 添加一个批次维度，形状变为 (1, C, H, W)
    img = img.to(device=device, dtype=torch.float32)    # 将张量移动到指定设备，并转换为浮点类型

    # 禁用梯度计算，以减少内存占用和加速推理
    with torch.no_grad():
        output = net(img).cpu()    # 前向传播，得到模型输出，并将结果移回CPU

        # 使用双线性插值将输出调整到与原始图像相同的大小
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')


        if net.n_classes > 1:    # 多分类分割
            mask = output.argmax(dim=1)    # 找到每个像素所属的类别
        else:    # 二分类分割
            mask = torch.sigmoid(output) > out_threshold    # 应用sigmoid并根据阈值生成二值掩膜

    return mask[0].long().squeeze().numpy()    # 返回预测的掩膜，并转换为numpy数组


def get_args():
    """解析命令行参数，用于设置模型加载路径、输入输出文件路径及其他配置。

        参数：无

        返回：
            - args: 包含所有解析参数的命名空间对象。
    """

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    # 添加参数：模型路径
    parser.add_argument('--pth', '-p', type=str, default='MODEL.pth', metavar='FILE',
                        help='Load model from a .pth file')

    # 添加参数：输入图像文件名
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)

    # 添加参数：输出图像文件名
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')

    # 添加参数：是否可视化处理过程
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')

    # 添加参数：是否不保存输出掩膜
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')

    # 添加参数：掩膜阈值
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')

    # 添加参数：输入图像的缩放因子
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    # 添加参数：输入图像的通道数
    parser.add_argument('--channels', '-ch', type=int, default=1, help='Number of channels in input images')

    # 添加参数：分类数量
    parser.add_argument('--classes', '-cl', type=int, default=2, help='Number of classes')

    # 添加参数：是否使用双线性上采样
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    # return parser.parse_args()    # 解析命令行参数并返回
    # 解析参数
    parsed_args = parser.parse_args()

    return parsed_args


def get_output_filenames(args):
    """生成输出文件名列表。

        如果未提供输出文件名，则基于输入文件名生成默认的输出文件名。
        默认的输出文件名格式为：<输入文件名>_OUT.png

        参数：
            - args: 包含输入和输出文件名信息的命名空间对象。

        返回：
            - 输出文件名列表。
    """
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'    # 去掉文件扩展名，并添加"_OUT.png"后缀

    return args.output or list(map(_generate_name, args.input))    # 如果提供了输出文件名，直接返回；否则，基于输入文件名生成默认输出文件名


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
    elif mask_values == [0, 1]:    # 如果mask_values是布尔值范围，初始化二值掩膜
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:    # 其他情况，初始化单通道的8位无符号整数数组
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # 如果掩膜是三维的，选择每个像素值最大的通道索引作为类别
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # 遍历每个类别索引和对应的像素值，将掩膜中该类别的像素赋值为指定值
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    # 将numpy数组转换为PIL.Image并返回
    return Image.fromarray(out)


if __name__ == '__main__':
    sys.argv = [
        'predict.py',
        '--pth', 'D:/xuniCpan/Graduation Design/graduationDesign/ml/netModels/unet.pth',
        '--input', 'D:/xuniCpan/Graduation Design/graduationDesign/ml/test_img/P19-0080.png',
        '--scale', '0.5',
        '--viz',
        '--no-save'
    ]
    args = get_args()    # 解析命令行参数

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')    # 配置日志输出格式

    in_files = args.input    # 输入文件列表

    out_files = get_output_filenames(args)    # 生成输出文件列表

    # 设置设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 加载模型参数
    logging.info(f'Loading model from {args.pth}')
    state_dict = torch.load(args.pth, map_location=device)
    model_name = state_dict.pop('model_name', None)    # 从模型参数中提取模型名称
    mask_values = state_dict.pop('mask_values', [0, 1])    # 从模型参数中提取掩膜值

    # 根据模型名称实例化对应的模型
    if model_name == 'unet++':
        net = UNetPlusPlus(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif model_name == 'u2net':
        net = U2Net(n_channels=args.channels, n_classes=args.classes)
    elif model_name == 'unet_cs':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=True, s_attention=True)
    elif model_name == 'unet_c':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=True, s_attention=False)
    elif model_name == 'unet_s':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=False, s_attention=True)
    elif model_name == 'unet':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=False, s_attention=False)
    else:
        raise ValueError(f'Model {model_name} not recognized')    # 如果模型名称无效，抛出异常

    net.to(device=device)    # 将模型移动到指定设备
    net.load_state_dict(state_dict)    # 加载模型参数

    logging.info(f'Model {model_name} loaded!')

    # 遍历输入图像文件，逐个进行分割预测
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')    # 输出当前处理的文件名
        img = Image.open(filename)   # 加载图像

        # 使用模型进行分割预测
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # 如果未指定 no_save 参数，则保存结果掩码文件
        if not args.no_save:
            out_filename = out_files[i]    # 获取当前输出文件名
            result = mask_to_image(mask, mask_values)    # 将预测的掩码转换为图像格式
            result.save(out_filename)    # 保存转换后的图像到指定路径
            logging.info(f'Mask saved to {out_filename}')    # 记录日志，提示掩码已保存

        # 如果指定了 viz 参数，则可视化图像和对应的掩码
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')   # 提示用户当前图像正在可视化
            plot_img_and_mask(img, mask)    # 显示原始图像及其预测掩码
