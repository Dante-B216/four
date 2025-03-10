# 导入PyTorch库
import torch  # PyTorch主库，提供各种张量操作和深度学习相关功能
import torch.nn.functional as F  # 提供用于神经网络的常用函数，如激活函数、损失函数等

# 导入tqdm库，用于显示进度条
from tqdm import tqdm  # tqdm用于显示训练或验证过程中进度条，使得过程更加可视化

# 导入自定义的评估指标函数
from ml.tools.dice_score import multiclass_dice_coeff, dice_coeff  # 导入计算Dice系数的函数
# multiclass_dice_coeff：用于多分类任务中计算Dice系数
# dice_coeff：用于二分类任务中计算Dice系数

from ml.tools.iou_score import multiclass_iou_coeff, iou_coeff  # 导入计算IoU（Intersection over Union）系数的函数
# multiclass_iou_coeff：用于多分类任务中计算IoU系数
# iou_coeff：用于二分类任务中计算IoU系数

from ml.tools.prec_score import precision  # 导入计算精确度（Precision）得分的函数

from ml.tools.acc_score import accuracy  # 导入计算准确度（Accuracy）得分的函数

# 定义评估函数，用于计算模型在验证集上的表现
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()    # 设置模型为评估模式，关闭dropout和batchnorm等训练时使用的操作
    num_val_batches = len(dataloader)    # 获取验证集的batch数量
    dice_score = 0    # 初始化Dice分数
    iou_score = 0    # 初始化IoU分数
    prec_score = 0    # 初始化精确度（Precision）分数
    acc_score = 0    # 初始化准确度（Accuracy）分数

    # 在自动混合精度模式下进行验证
    with torch.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=amp):
        # 迭代验证集中的每个batch
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']    # 获取当前batch的图像和真实标签

            # 将图片和标签移动到指定的设备（如GPU），并转换为合适的数据类型
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 使用模型进行预测，得到预测的mask
            mask_pred = net(image)

            # 如果是二分类任务（只有背景和前景）
            if net.n_classes == 1:
                # 确保真实标签是0或1
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

                # 使用Sigmoid激活函数，将预测值转为概率，并进行二值化处理（大于0.5为前景，小于0.5为背景）
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # 计算Dice系数
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                # 计算IoU系数
                iou_score += iou_coeff(mask_pred, mask_true, reduce_batch_first=False)

                # 计算精确度（Precision）
                prec_score += precision(mask_pred, mask_true)

                # 计算准确度（Accuracy）
                acc_score += accuracy(mask_pred, mask_true)
            else:
                # 如果是多分类任务，确保真实标签在类别范围内
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # 将真实标签转换为one-hot格式
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                # 将预测结果转换为one-hot格式，并选择预测类别最大的值
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # 计算多分类任务中的Dice系数，忽略背景类（索引从1开始）
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                # 计算多分类任务中的IoU系数，忽略背景类
                iou_score += multiclass_iou_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                # 计算多分类任务中的精确度（Precision），忽略背景类
                prec_score += precision(mask_pred[:, 1:], mask_true[:, 1:])
                # 计算多分类任务中的准确度（Accuracy），忽略背景类
                acc_score += accuracy(mask_pred[:, 1:], mask_true[:, 1:])

    net.train()    # 恢复模型为训练模式

    # 返回平均评分（各个指标的总分除以验证集的batch数量）
    return dice_score / max(num_val_batches, 1), iou_score / max(num_val_batches, 1), \
        prec_score / max(num_val_batches, 1), acc_score / max(num_val_batches, 1)
