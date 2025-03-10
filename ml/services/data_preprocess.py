# Please manually organize dataset files into the structure as follows before you run the data preprocess script:
# ../TrainingSet
# ../TestSet/
#   - /Test1Set
#   - /Test2Set
#   - /Test1SetContours
#   - /Test2SetContours

# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能，如文件路径操作
import shutil  # 用于文件的复制、移动和删除操作
import pydicom  # 用于读取和处理 DICOM 格式的医学图像文件
from PIL import Image  # Python Imaging Library，用于图像的处理和保存
import numpy as np  # 用于处理数组和执行数值计算
import cv2  # OpenCV 库，用于图像处理和计算机视觉
from albumentations import (  # 图像增强库，支持多种数据增强方法
    Compose,  # 用于组合多个数据增强操作
    HorizontalFlip,  # 随机水平翻转
    VerticalFlip,  # 随机垂直翻转
    ShiftScaleRotate,  # 随机平移、缩放和旋转
    RandomBrightnessContrast,  # 随机调整亮度和对比度
    GaussNoise,  # 添加高斯噪声
    ElasticTransform,  # 弹性形变
    RandomResizedCrop  # 随机裁剪并调整大小
)
from tqdm import tqdm  # 进度条库，用于显示操作进度
from pathlib import Path  # 提供面向对象的路径操作


# 确保指定目录存在，如果不存在，则创建该目录
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 将 DICOM 文件转换为 PNG 图像
def convert_dcm_to_png(dicom_path, png_path):
    # pydicom.dcmread(dicom_path) 读取 DICOM 文件
    dcm_image = pydicom.dcmread(dicom_path).pixel_array

    # 将 DICOM 图像数据转换为 PIL 图像对象
    im = Image.fromarray(dcm_image)

    # 转换为灰度图像 (L模式)，因为 DICOM 图像通常是灰度的
    im = im.convert('L')

    # 将图像保存为 PNG 格式
    im.save(png_path)

# 根据轮廓文件创建一个掩膜图像
def create_mask_from_contour(contour_path, mask_path, image_shape):
    # 初始化一个列表用于存储轮廓点
    contours = []

    # 打开轮廓文件并读取其中的点
    with open(contour_path, 'r') as file:
        contour = []  # 存储单个轮廓的点
        for line in file:
            # 将文件中的点转换为整数，并四舍五入
            x, y = map(lambda v: int(round(float(v))), line.split())
            contour.append([x, y])  # 将点添加到当前轮廓中
        # 将单个轮廓转换为 NumPy 数组并添加到轮廓列表
        contours.append(np.array(contour, dtype=np.int32))

    # 创建一个全黑（零值）的掩膜，尺寸与指定的图像形状相同
    mask = np.zeros(image_shape, dtype=np.uint8)

    # 使用 OpenCV 的 fillPoly 函数在掩膜中填充轮廓内部区域为白色
    # contours：轮廓点列表，255：填充的灰度值（白色）
    cv2.fillPoly(mask, contours, 255)

    # 将生成的掩膜转换为图像并保存到指定路径
    Image.fromarray(mask).save(mask_path)


# 创建目标文件夹路径
def copy_and_process_files(base_dir, dataset_folder, target_folder, start_index, end_index):
    target_dir_imgs = os.path.join(base_dir, target_folder, 'imgs')     # 存储图片的文件夹
    target_dir_i_masks = os.path.join(base_dir, target_folder, 'i-masks')  # 存储内层轮廓掩膜的文件夹
    target_dir_o_masks = os.path.join(base_dir, target_folder, 'o-masks')  # 存储外层轮廓掩膜的文件夹

    # 确保目标文件夹存在，如果不存在，则创建
    ensure_directory_exists(target_dir_imgs)
    ensure_directory_exists(target_dir_i_masks)
    ensure_directory_exists(target_dir_o_masks)

    # # 遍历指定范围的患者文件夹
    for i in range(start_index, end_index + 1):

        # 构建患者文件夹的名称
        patient_folder = f'patient{i:02d}'

        # 构建患者对应的列表文件路径
        list_file_path = os.path.join(base_dir, dataset_folder, patient_folder, f'P{i:02d}list.txt')

        # 如果列表文件不存在，则跳过当前患者
        if not os.path.exists(list_file_path):
            print(f"List file not found: {list_file_path}")
            continue

        # # 读取并处理每一行（每个文件路径）
        with open(list_file_path, 'r') as file:
            for line in file:
                line = line.strip().replace('.\\', '').replace('\\', '/')       # 清理路径格式，统一使用正斜杠
                contour_filename = line.split('/')[-1]      # 获取轮廓文件的名称

                # # 处理 DICOM 文件并将其保存为 PNG 图像
                # 将轮廓文件名转换为对应的 DICOM 文件名
                dicom_filename = contour_filename.replace('-icontour-manual.txt', '.dcm').replace('-ocontour-manual.txt', '.dcm')

                # 构建 DICOM 文件的路径
                dicom_path = os.path.join(base_dir, dataset_folder, patient_folder, f'P{i:02d}dicom', dicom_filename)

                # 构建 PNG 文件的保存路径
                png_filename = dicom_filename.replace('.dcm', '.png')
                png_path = os.path.join(target_dir_imgs, png_filename)

                # 如果 DICOM 文件存在，进行转换并保存为 PNG
                if os.path.exists(dicom_path):
                    convert_dcm_to_png(dicom_path, png_path)
                    print(f"Converted and saved: {png_path}")
                else:
                    print(f"Dicom file not found: {dicom_path}")
                    continue  # If the DICOM file does not exist, skip subsequent processing

                #  # 根据轮廓文件创建并保存掩膜
                # 判断是内层轮廓（i-contour）还是外层轮廓（o-contour）
                mask_path = os.path.join(target_dir_i_masks if 'icontour' in contour_filename else target_dir_o_masks, png_filename)

                # 构建轮廓文件路径
                contour_path = os.path.join(base_dir, dataset_folder, line)

                # 如果轮廓文件和 PNG 文件都存在，创建掩膜
                if os.path.exists(contour_path) and os.path.exists(png_path):
                    image_shape = Image.open(png_path).size[::-1]       # 获取 PNG 图像的尺寸（高, 宽）
                    create_mask_from_contour(contour_path, mask_path, image_shape)      # 调用创建掩膜的函数
                    print(f"Created and saved mask: {mask_path}")
                else:
                    print(f"Contour file or PNG file not found: {contour_path} or {png_path}")


def copy_contours(base_dir, dataset_folder, source_folder, target_folder, start_index, end_index):
    # 构建源目录和目标目录的路径
    source_dir = os.path.join(base_dir, dataset_folder, source_folder)    # 源文件夹路径
    target_base_dir = os.path.join(base_dir, dataset_folder, target_folder)    # 目标文件夹的基础路径

    # # 确保源目录和目标基础目录存在
    if not os.path.exists(source_dir) or not os.path.exists(target_base_dir):
        print("Source or target base directory does not exist.")
        return    # 如果源目录或目标基础目录不存在，打印错误信息并返回

    # 遍历指定范围内的文件夹（根据 start_index 和 end_index）
    for i in range(start_index, end_index + 1):

        # 构建源患者文件夹路径
        source_folder = os.path.join(source_dir, f"P{i:02d}contours-manual")

        # 构建目标患者文件夹路径
        target_folder = os.path.join(target_base_dir, f"patient{i:02d}")

        # 检查源文件夹是否存在
        if not os.path.exists(source_folder):
            print(f"Source folder not found: {source_folder}")
            continue    # 如果源文件夹不存在，跳过该患者

        # 如果目标文件夹不存在，则创建目标文件夹
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 将源文件夹复制到目标文件夹中
        shutil.copytree(source_folder, os.path.join(target_folder, f"P{i:02d}contours-manual"), dirs_exist_ok=True)
        print(f"Copied {source_folder} to {target_folder}")


def augment_data(img_path, mask_i_path, mask_o_path, output_dir, image_name, transform, times):
    # # 读取图像和掩码
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)    # 读取输入图像为灰度图
    mask_i = cv2.imread(str(mask_i_path), cv2.IMREAD_GRAYSCALE)    # 读取内层掩码（Endocardial contour）
    mask_o = cv2.imread(str(mask_o_path), cv2.IMREAD_GRAYSCALE)    # 读取外层掩码（Epicardial contour）

    # # 保存原始图像和掩码
    cv2.imwrite(str(output_dir / 'imgs' / image_name), image)    # 保存原始图像
    cv2.imwrite(str(output_dir / 'i-masks' / image_name), mask_i)    # 保存原始内层掩码
    cv2.imwrite(str(output_dir / 'o-masks' / image_name), mask_o)    # 保存原始外层掩码

    # 执行数据增强，并保存多个增强后的图像和掩码
    for i in range(times):
        # # 对图像和掩码应用增强
        augmented = transform(image=image, masks=[mask_i, mask_o])    # 使用albumentations库进行图像和掩码的增强
        image_aug, mask_i_aug, mask_o_aug = augmented['image'], augmented['masks'][0], augmented['masks'][1]    # 提取增强后的图像和掩码

        # 保存增强后的图像和掩码
        cv2.imwrite(str(output_dir / 'imgs' / f'aug{i}_{image_name}'), image_aug)    # 保存增强后的图像
        cv2.imwrite(str(output_dir / 'i-masks' / f'aug{i}_{image_name}'), mask_i_aug)    # 保存增强后的内层掩码
        cv2.imwrite(str(output_dir / 'o-masks' / f'aug{i}_{image_name}'), mask_o_aug)    # 保存增强后的外层掩码


def augmentation(img_dir, mask_i_dir, mask_o_dir, output_dir, times):
    # 确保输出目录及其子目录存在，如果不存在则创建
    output_dir.mkdir(parents=True, exist_ok=True)    # 创建输出目录
    (output_dir / 'imgs').mkdir(parents=True, exist_ok=True)    # 创建存放图像的子目录
    (output_dir / 'i-masks').mkdir(parents=True, exist_ok=True)    # 创建存放内层掩码的子目录
    (output_dir / 'o-masks').mkdir(parents=True, exist_ok=True)    # 创建存放外层掩码的子目录

    # 定义图像增强的操作，包括多种常见的图像处理方法
    transform = Compose([
        HorizontalFlip(p=0.5),    # 以50%的概率进行水平翻转
        VerticalFlip(p=0.5),    # 以50%的概率进行垂直翻转
        ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=45, p=0.5),    # 平移、缩放、旋转
        RandomBrightnessContrast(p=0.2),    # 随机调整亮度和对比度
        GaussNoise(p=0.2),    # 添加高斯噪声
        ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),    # 弹性变换
        RandomResizedCrop(height=256, width=216, scale=(0.3, 1.0), p=0.5)    # 随机裁剪并调整尺寸
    ])

    # 遍历输入图像文件夹中的所有图像文件，进行增强
    for img_name in tqdm(os.listdir(img_dir), desc='Augmenting images'):
        if img_name.endswith('.png'):    # 仅处理以 .png 结尾的文件
            # 构造图像及其对应掩码的路径
            img_path = img_dir / img_name    # 图像文件的路径
            mask_i_path = mask_i_dir / img_name    # 内层掩码文件的路径
            mask_o_path = mask_o_dir / img_name    # 外层掩码文件的路径

            # 调用 augment_data 函数进行数据增强
            augment_data(img_path, mask_i_path, mask_o_path, output_dir, img_name, transform, times)


def get_args():
    # 创建 ArgumentParser 对象，描述命令行工具的功能
    parser = argparse.ArgumentParser(description='Data preprocessing and augmentation')

    # 添加命令行参数 --times 或 -t，用于指定数据增强的次数
    parser.add_argument('--times', '-t', type=int, default=4, help='Augmentation times')

    # 解析并返回命令行参数
    return parser.parse_args()


if __name__ == '__main__':
    # 获取命令行参数，主要是数据增强的次数
    # args = get_args()

    # 手动设置args字典，模拟命令行参数
    args = {
        'times': 4,  # 例如设置数据增强的次数为4
    }

    # 设置基础目录路径，表示数据所在的根目录
    base_dir = '.'

    # 调用函数复制并处理训练集数据，将文件从源位置复制到目标文件夹
    copy_and_process_files(base_dir, 'TrainingSet', 'train_data', 1, 16)

    # 调用函数复制并处理测试集1的轮廓数据
    copy_contours(base_dir, 'TestSet', "Test1SetContours", "Test1Set", 17, 32)

    # 调用函数复制并处理测试集2的轮廓数据
    copy_contours(base_dir, 'TestSet', "Test2SetContours", "Test2Set", 33, 48)

    # 调用函数复制并处理测试集1的图像和掩膜数据
    copy_and_process_files(base_dir, 'TestSet/Test1Set', 'test1_data', 17, 32)

    # 调用函数复制并处理测试集2的图像和掩膜数据
    copy_and_process_files(base_dir, 'TestSet/Test2Set', 'test2_data', 33, 48)

    # 打印文件复制和处理完成的提示信息
    print("Copying and processing files finished.")

    # 从args字典中获取数据增强的次数
    times = args['times']

    # 调用数据增强函数，对训练集的图像和掩膜进行增强操作
    augmentation(Path('./train_data/imgs'), Path('./train_data/i-masks'), Path('./train_data/o-masks'),
                 Path('./train_data_aug'), times)

    # 打印数据增强完成的提示信息
    print("Data augmentation finished.")