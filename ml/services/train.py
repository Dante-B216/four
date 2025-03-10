# 导入必要的库
import argparse  # 用于解析命令行参数
import logging  # 用于记录日志
import os  # 提供与操作系统交互的功能
import torch  # PyTorch框架的主库
import torch.nn as nn  # 包含神经网络模块
import torch.nn.functional as F  # 包含功能性神经网络操作
from pathlib import Path  # 用于处理文件和路径
from torch import optim  # 包含优化器
from torch.utils.data import DataLoader, random_split  # 数据加载和分割工具
from tqdm import tqdm  # 用于创建进度条

# 导入第三方库
import wandb  # 用于实验跟踪和可视化
from evaluate import evaluate  # 用于验证模型性能的自定义模块
from ml.netModelsTools import UNet, UNetPlusPlus, U2Net  # 导入多种模型
from ml.tools.data_loading import BasicDataset  # 数据集加载类
from ml.tools.dice_score import dice_loss  # 自定义的Dice损失函数



def train_model(
        model,  # 要训练的神经网络模型
        device,  # 模型和数据加载的设备（CPU或GPU）
        dir_img: Path,  # 包含训练图像的目录
        dir_mask: Path,  # 包含对应分割掩码的目录
        dir_checkpoint: Path,  # 在训练期间保存模型检查点的目录
        epochs: int,  # 训练的总轮次
        batch_size: int,  # 每批数据的样本数
        learning_rate: float,  # 优化器的学习率
        val_percent: float,  # 数据集中用于验证的比例
        save_checkpoint: bool,  # 是否保存模型检查点
        img_scale: float,  # 图像的缩放比例
        amp: bool,  # 是否使用自动混合精度（AMP）进行训练
        weight_decay: float,  # 权重衰减（L2正则化）系数
        momentum: float,  # 优化器的动量参数
        gradient_clipping: float,  # 梯度裁剪的最大范数
        epochs_per_checkpoint: int,  # 每隔多少轮保存一次检查点
        loss_function: str,  # 损失函数类型（'dice'、'ce'或'dice+ce'）
        optimizer_name: str,  # 优化器的名称（'adam'或'rmsprop'）
):
    # 1. 从图像和掩码目录创建数据集
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. 将数据集划分为训练集和验证集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 初始化日志记录
    experiment = wandb.init(project='RVSC', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    # 打印训练参数
    logging.info(f'''Starting training:
        Model:           {model.model_name}
        Channels:        {model.n_channels}
        Classes:         {model.n_classes}
        Bilinear:        {model.bilinear}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed precision: {amp}
        Weight decay:    {weight_decay}
        Momentum:        {momentum}
        Gradient clipping: {gradient_clipping}
        Epochs per checkpoint: {epochs_per_checkpoint}
        Loss function:   {loss_function}
        Optimizer:       {optimizer_name}
        
    ''')

    # 4. 设置优化器、损失函数、学习率调度器和AMP的梯度缩放器
    if optimizer_name == 'rmsprop':    # 使用RMSProp优化器
        optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    else:   # 默认使用Adam优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)

    # 学习率调度器：当验证指标未改善时降低学习率
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, factor=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=100)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)    # 自动混合精度梯度缩放
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()    # 设置损失函数
    global_step = 0    # 全局训练步数

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()    # 设置模型为训练模式
        epoch_loss = 0    # 初始化每轮损失
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']    # 获取图像和对应的掩码

                # 确保输入图像的通道数与模型定义的输入通道数匹配
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # 将图像和掩码加载到设备上
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 使用自动混合精度加速训练
                with torch.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=amp):
                # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)      # 模型预测

                    # 根据损失函数类型计算损失
                    if model.n_classes == 1:
                        if loss_function == 'dice':
                            loss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        elif loss_function == 'ce':
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        else:   # loss_function == 'dice+ce'
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        if loss_function == 'dice':
                            loss = dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                        elif loss_function == 'ce':
                            loss = criterion(masks_pred, true_masks)
                        else:   # loss_function == 'dice+ce'
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                # 梯度计算和优化步骤
                optimizer.zero_grad(set_to_none=True)       # 清零梯度
                grad_scaler.scale(loss).backward()      # 反向传播，使用梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)       # 梯度裁剪
                grad_scaler.step(optimizer)     # 更新模型参数
                grad_scaler.update()        # 更新缩放因子

                # 更新进度条和日志
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # # 每隔一定步数执行验证
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}    # 记录模型权重和梯度的直方图
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # 在验证集上评估模型性能
                        dice_score, iou_score, prec_score, acc_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(dice_score)

                        # 记录验证分数到日志
                        logging.info('Validation Dice score: {}'.format(dice_score))
                        logging.info('Validation Iou score: {}'.format(iou_score))
                        logging.info('Validation Precision score: {}'.format(prec_score))
                        logging.info('Validation Accuracy score: {}'.format(acc_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': dice_score,
                                'validation Iou': iou_score,
                                'validation Precision': prec_score,
                                'validation Accuracy': acc_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        # 定期保存模型检查点
        if save_checkpoint and epoch % epochs_per_checkpoint == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['model_name'] = model.model_name
            state_dict['mask_values'] = dataset.mask_values
            # torch.save(state_dict, str(dir_checkpoint / f'{model.model_name}_checkpoint_epoch{epoch}.pth'))

            # Ensure dir_checkpoint is a Path object
            dir_checkpoint = Path(dir_checkpoint)

            # Now, this should work
            torch.save(state_dict, str(dir_checkpoint / f'{model.model_name}_checkpoint_epoch{epoch}.pth'))

            logging.info(f'Checkpoint {epoch} saved!')

    wandb.finish()    # 结束日志记录


def get_args():
    # 创建一个参数解析器，用于从命令行获取参数
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    # 模型名称参数
    parser.add_argument('--model', '-md', metavar='M', type=str, default='unet',
                        help='Name of model ("unet", "unet_c", "unet_s", "unet_cs", "unet++", "u2net")')

    # 输入图像通道数参数
    parser.add_argument('--channels', '-ch', type=int, default=1, help='Number of channels in input images')

    # 输出类别数参数
    parser.add_argument('--classes', '-cl', type=int, default=2, help='Number of classes')

    # 是否使用双线性插值进行上采样
    parser.add_argument('--bilinear', '-bl', action='store_true', default=False, help='Use bilinear upsampling')

    # 输入图像文件夹路径
    parser.add_argument('--imgs', '-i', metavar='IMGS', type=str, default='../train_data_aug/imgs/',
                        help='directory of input images')

    # 分割掩码文件夹路径
    parser.add_argument('--masks', '-mx', metavar='MASK', type=str, default='../train_data_aug/i-masks/',
                        help='directory of target masks')

    # 保存检查点文件夹路径
    parser.add_argument('--save', '-sv', metavar='SAVE', type=str, default='../i-checkpoints/',
                        help='directory of saved model checkpoints')

    # 训练轮次参数
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')

    # 批量大小参数
    parser.add_argument('--batch-size', '-bs', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')

    # 学习率参数
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')

    # 从.pth文件加载模型参数
    parser.add_argument('--pth', '-p', type=str, default=False, help='Load model from a .pth file')

    # 图像缩放比例参数
    parser.add_argument('--scale', '-sc', type=float, default=0.5, help='Downscaling factor of the images')

    # 验证集比例参数
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    # 是否使用混合精度训练
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    # 权重衰减参数
    parser.add_argument('--weight-decay', '-w', type=float, default=1e-8, help='Weight decay')

    # 动量参数
    parser.add_argument('--momentum', '-mm', type=float, default=0.999, help='Momentum')

    # 梯度裁剪最大范数
    parser.add_argument('--gradient-clipping', '-gc', type=float, default=1.0, help='Gradient clipping')

    # 保存检查点的轮次间隔
    parser.add_argument('--epochs-per-checkpoint', '-epc', type=int, default=1, help='Save checkpoint every N epochs')

    # 损失函数类型参数
    parser.add_argument('--loss', '-ls', type=str, default='dice+ce', help='Loss function ("dice", "ce", "dice+ce")')

    # 优化器类型参数
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Optimizer ("adam", "rmsprop")')

    # 解析参数
    parsed_args = parser.parse_args()

    # 对模型名称参数进行有效性验证
    assert parsed_args.model in ['unet', 'unet_c', 'unet_s', 'unet_cs', 'unet++', 'u2net']
    # 对损失函数参数进行有效性验证
    assert parsed_args.loss in ['dice', 'ce', 'dice+ce']
    # 对优化器参数进行有效性验证
    assert parsed_args.optimizer in ['adam', 'rmsprop']

    # 返回解析后的参数
    return parsed_args


if __name__ == '__main__':
    # 获取命令行参数
    #args = get_args()

    # 定义训练参数，直接在代码中输入
    args = argparse.Namespace(
        model='u2net',  # 模型名称，可选值："unet", "unet_c", "unet_s", "unet_cs", "unet++", " u2net"
        channels=1,  # 输入图像的通道数（如1表示灰度图像，3表示RGB图像）
        classes=2,  # 输出的类别数
        bilinear=False,  # 是否使用双线性上采样
        imgs='./train_data_aug/imgs/',  # 输入图像目录
        masks='./train_data_aug/i-masks/',  # 分割掩码目录
        save='./i-checkpoints/',  # 检查点保存目录
        epochs=50,  # 训练轮数
        batch_size=32,  # 每个批次的样本数量
        lr=1e-5,  # 学习率
        pth=None,  # 预训练模型文件路径（如果有）
        scale=0.5,  # 图像缩放比例
        val=10.0,  # 验证集占比（百分比）
        amp=False,  # 是否启用混合精度训练
        weight_decay=1e-3,  # 权重衰减系数
        momentum=0.999,  # 动量
        gradient_clipping=1.0,  # 梯度裁剪值
        epochs_per_checkpoint=1,  # 每隔多少轮保存一次检查点
        loss='dice+ce',  # 损失函数类型
        optimizer='adam'  # 优化器类型
    )

    # 配置日志记录，设置日志级别为INFO，指定日志输出格式
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 检测设备是否支持GPU，如果支持则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 根据命令行参数选择和初始化模型
    # args.model指定了模型类型，根据不同的模型类型初始化相应的模型
    # 每种模型根据输入通道数(n_channels)、输出类别数(n_classes)和是否使用双线性上采样(bilinear)进行配置
    if args.model == 'unet++':
        model = UNetPlusPlus(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'u2net':
        model = U2Net(n_channels=args.channels, n_classes=args.classes)
    elif args.model == 'unet_cs':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=True, s_attention=True)
    elif args.model == 'unet_c':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=True, s_attention=False)
    elif args.model == 'unet_s':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=False, s_attention=True)
    else:   # 默认为UNet模型
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=False, s_attention=False)

    # 设置模型内存格式为"channels_last"以提高内存和计算效率
    model = model.to(memory_format=torch.channels_last)

    # 输出模型的基本配置信息
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 如果指定了从.pth文件加载模型权重
    if args.pth:
        # 加载权重文件并将其映射到当前设备
        state_dict = torch.load(args.pth, map_location=device)

        # 确保加载的模型类型与命令行参数指定的一致
        assert args.model == state_dict.pop('model_name', None)

        # 移除权重文件中不必要的键
        state_dict.pop('mask_values', [0, 1])

        # 将权重加载到模型中
        model.load_state_dict(state_dict)

        # 记录权重加载成功信息
        logging.info(f'Model loaded from {args.load}')

    # 将模型转移到目标设备（GPU或CPU）
    model.to(device=device)

    # 尝试开始训练模型
    try:
        train_model(
            model=model,
            device=device,
            dir_img=args.imgs,
            dir_mask=args.masks,
            dir_checkpoint=args.save,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            save_checkpoint=True,
            img_scale=args.scale,
            amp=args.amp,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            epochs_per_checkpoint=args.epochs_per_checkpoint,
            loss_function=args.loss,
            optimizer_name=args.optimizer
        )
    # 捕获显存不足的异常并处理
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        # 清空GPU缓存以释放显存
        torch.cuda.empty_cache()

        # 启用模型检查点机制以降低显存需求
        model.use_checkpointing()

        # 再次尝试训练模型
        train_model(
            model=model,
            device=device,
            dir_img=args.imgs,
            dir_mask=args.masks,
            dir_checkpoint=args.save,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            save_checkpoint=True,
            img_scale=args.scale,
            amp=args.amp,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            epochs_per_checkpoint=args.epochs_per_checkpoint,
            loss_function=args.loss,
            optimizer_name=args.optimizer
        )