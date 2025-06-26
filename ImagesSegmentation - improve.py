import os
import csv
import torch
import numpy as np
import scipy.io
from PIL import Image
import torchvision.transforms as transforms
import time

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode


def load_color_and_names():
    """
    加载颜色和类别名称信息
    :return: 颜色信息和类别名称字典
    """
    colors = scipy.io.loadmat('data/color150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    return colors, names


def build_segmentation_module():
    """
    构建并加载分割模型
    :return: 分割模型
    """
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()
    return segmentation_module


def preprocess_image(pil_image):
    """
    对图像进行预处理
    :param pil_image: PIL图像
    :return: 预处理后的图像数据
    """
    pil_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    img_data = pil_to_tensor(pil_image)
    return img_data


def visualize_result(img, pred, colors, index=None):
    """
    可视化分割结果
    :param img: 原始图像
    :param pred: 分割预测结果
    :param colors: 颜色信息
    :param index: 类别索引
    :return: 可视化后的图像
    """
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    return pred_color


def process_image(image_path, segmentation_module, colors, id_save):
    """
    处理单张图像
    :param image_path: 图像路径
    :param segmentation_module: 分割模型
    :param colors: 颜色信息
    :param id_save: 保留的类别数量
    :return: 图像名称和各类别像素占比列表
    """
    try:
        pil_image = Image.open(image_path).convert('RGB')
        img_original = np.array(pil_image)
        img_data = preprocess_image(pil_image)
        singleton_batch = {'img_data': img_data[None].cuda()}
        output_size = img_data.shape[1:]

        with torch.no_grad():
            scores = segmentation_module(singleton_batch, segSize=output_size)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        vs_total = visualize_result(img_original, pred, colors)
        image_name = os.path.basename(image_path)
        rate_list = [image_name]
        for n in range(id_save):
            vs_single = visualize_result(img_original, pred, colors, n)
            num = np.count_nonzero((vs_single != [0, 0, 0]).all(axis=2))
            all_num = vs_single.shape[0] * vs_single.shape[1]
            rate = num / all_num
            rate_list.append(rate)

        torch.cuda.empty_cache()  # 释放GPU缓存
        return image_name, vs_total, rate_list
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None, None, None


def main():
    start_time = time.time()

    # 加载颜色和类别信息
    colors, names = load_color_and_names()

    # 构建分割模型
    segmentation_module = build_segmentation_module()

    # 写入街景图片储存位置
    images_storage_path = r'C:\Users\liwen\Desktop\LWH StreetView\03 ImagesSegmentation\images'
    # 分割后结果图片保存位置
    image_result_path = r'C:\Users\liwen\Desktop\LWH StreetView\03 ImagesSegmentation\images_result'
    # 分割后结果csv保存位置
    image_result_csv = r'C:\Users\liwen\Desktop\LWH StreetView\03 ImagesSegmentation/segmentation_result.csv'
    # 设置保留前多少位id结果
    id_save = 150

    # 设置表头
    headers = ['id']
    for id_num in range(id_save):
        head = f'id{id_num + 1}'
        headers.append(head)

    # 创建csv文件及表头
    with open(image_result_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    # 遍历图片文件夹
    for root, dirs, files in os.walk(images_storage_path):
        for image_name in files:
            image_path = os.path.join(root, image_name)
            image_name, vs_total, rate_list = process_image(image_path, segmentation_module, colors, id_save)
            if image_name and vs_total and rate_list:
                # 保存分割后图像
                image = Image.fromarray(vs_total)
                image.save(os.path.join(image_result_path, image_name))

                # 将结果写入csv
                with open(image_result_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(rate_list)

                print(' [-] 处理完成图片：', image_path)

    end_time = time.time()
    print(' [-] 处理完成所有图片,耗时{:.2f}秒'.format(end_time - start_time))


if __name__ == "__main__":
    main()