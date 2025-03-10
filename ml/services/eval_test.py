# 导入命令行解析库
import argparse  # argparse用于处理命令行参数，方便在运行脚本时指定参数

# 导入日志记录库
import logging  # logging提供了一种灵活的日志记录方式，帮助跟踪代码的执行过程，特别是在调试时
import sys

# 导入PyTorch库
import torch  # PyTorch主库，提供张量操作、自动求导、神经网络组件等功能
from torch.utils.data import DataLoader  # DataLoader用于批量加载数据集，支持多线程加载和数据增强

# 导入自定义评估函数
from evaluate import evaluate  # 导入评估函数，用于计算模型在验证集上的各种评估指标

# 导入自定义模型架构
from ml.netModelsTools import UNet  # 导入UNet模型，这是常用于图像分割任务的网络架构
from ml.netModelsTools import UNetPlusPlus  # 导入UNet++模型，它是UNet的改进版，通过引入更多的跳跃连接增强模型表现
from ml.netModelsTools import U2Net  # 导入U2Net模型，专门设计用于边缘检测任务，具有更强的特征提取能力

# 导入自定义数据加载模块
from ml.tools.data_loading import BasicDataset  # 导入BasicDataset类，它用于加载和处理数据集，通常是继承自PyTorch的Dataset类

# 定义get_args()函数，用于解析命令行参数
def get_args():
    # 创建ArgumentParser对象，提供描述信息，方便用户了解脚本的用途
    parser = argparse.ArgumentParser(description='Test the model on images and target masks')

    # 添加参数：--pth/-p 用于指定加载的模型文件路径（默认为' MODEL.pth'）
    parser.add_argument('--pth', '-p', type=str, default='MODEL.pth', metavar='FILE',
                        help='Load model from a .pth file')

    # 添加参数：--input/-i 用于指定输入图像所在的目录
    parser.add_argument('--input', '-i', metavar='INPUT', help='Directory of input images')

    # 添加参数：--output/-o 用于指定输出图像（预测结果）保存的目录
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Directory of ouput masks')

    # 添加参数：--batch-size/-b 用于指定批处理大小，默认为1
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')

    # 添加参数：--scale/-s 用于指定图像的下采样因子，默认为0.5
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')

    # 添加参数：--channels/-ch 用于指定输入图像的通道数，默认为1（单通道图像）
    parser.add_argument('--channels', '-ch', type=int, default=1, help='Number of channels in input images')

    # 添加参数：--classes/-cl 用于指定模型的输出类别数，默认为2（背景和前景）
    parser.add_argument('--classes', '-cl', type=int, default=2, help='Number of classes')

    # 添加参数：--bilinear 用于指定是否使用双线性上采样，默认不使用
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    # 解析并返回命令行参数
    return parser.parse_args()

# 定义test_net()函数，用于测试模型在测试集上的表现
def test_net(net, device, test_loader, amp=False):
    # 将网络模型设置为评估模式，在此模式下，模型不会进行反向传播，并且禁用dropout等训练时特有的操作
    net.eval()

    # 调用evaluate函数，传入网络、测试数据加载器、设备和是否启用自动混合精度（amp）
    # 该函数会返回四个评估指标：Dice系数、IoU、精度（Precision）、准确度（Accuracy）
    dice_score, iou_score, prec_score, acc_score = evaluate(net, test_loader, device, amp)

    # 返回四个评估指标的值
    return dice_score, iou_score, prec_score, acc_score


if __name__ == '__main__':

    sys.argv = [
        'predict.py',
        '--pth', './i-checkpoints/unet_checkpoint_epoch5.pth',
        '--input', './test1_data/imgs/',
        '--output', './test1_data/i-masks',
        '--scale','0.5'
    ]

    # 解析命令行参数
    args = get_args()

    # 判断是否有可用的GPU，若没有则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置日志记录，设置日志级别为INFO，输出格式为：日志级别: 日志信息
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 输出正在使用的设备信息（GPU 或 CPU）
    logging.info(f'Using device {device}')

    # 输出加载模型的路径信息
    logging.info(f'Loading model from {args.pth}')

    # 加载模型权重（state_dict）并将其映射到指定的设备上（GPU或CPU）
    state_dict = torch.load(args.pth, map_location=device)

    # 从state_dict中弹出'模型名称'（如果存在），并将其赋值给model_name
    model_name = state_dict.pop('model_name', None)

    # 从state_dict中弹出'mask_values'（如果存在），如果没有则使用[0, 1]作为默认值
    mask_values = state_dict.pop('mask_values', [0, 1])

    # 根据从模型权重中提取的模型名称选择合适的网络架构
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
        raise ValueError(f'Model {model_name} not recognized')

    # 将网络模型移动到指定的设备（GPU或CPU）
    net.to(device=device)

    # 加载网络权重到模型中
    net.load_state_dict(state_dict)

    # 输出模型加载成功的信息
    logging.info(f'Model {model_name} loaded!')

    # 创建测试数据集实例，传入输入图像目录、输出目录和下采样因子
    test_dataset = BasicDataset(args.input, args.output, args.scale)

    # 创建测试数据加载器，批量大小为用户指定的batch_size，数据不打乱（用于评估）
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 调用test_net函数对模型进行评估，传入网络、设备和测试数据加载器
    dice, iou, prec, acc = test_net(net=net, device=device, test_loader=test_loader)

    # 输出评估结果（Dice系数、IoU、精度、准确度）
    logging.info(f'Dice score: {dice}')
    logging.info(f'Iou score: {iou}')
    logging.info(f'Precision: {prec}')
    logging.info(f'Accuracy: {acc}')
