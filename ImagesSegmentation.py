# https://github.com/CSAILVision/sceneparsing
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
import time

from IPython.display import display
import numpy as np
import cv2
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

start_time = time.time()
# 获取分割后的图片
def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
    #         print(f'{names[index+1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    # im_vis = numpy.concatenate((img, pred_color), axis=1)
    # display(PIL.Image.fromarray(im_vis))
    return pred_color

# 读取颜色文件
colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

# Network Builders
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
# segmentation_module.cpu()
segmentation_module.cuda()
# print(segmentation_module.eval())
# print(segmentation_module.cpu())

# Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])])  # across a large photo dataset.

# 写入街景图片储存位置
images_storage_path = r'C:\Users\liwen\Desktop\LWH StreetView\03 ImagesSegmentation\images'
# 分割后结果图片保存位置
image_result_path = r'C:\Users\liwen\Desktop\LWH StreetView\03 ImagesSegmentation\images_result'
# 分割后结果csv保存位置
image_result_csv = r'C:\Users\liwen\Desktop\LWH StreetView\03 ImagesSegmentation/segmentation_result.csv'
# 设置保留前多少位id结果
id_save = 150
# id_save = 151

# 设置表头
headers = ['id',]
for id_num in range(id_save):
    head = 'id%d'%(id_num+1)
    headers.append(head)
# 创建csv文件及表头
with open('%s'%image_result_csv ,'w' ,newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)


for i,j,k in os.walk(images_storage_path):
    for image_name in k:
        image_name_path = '{}/{}'.format(i,image_name)

        pil_image = PIL.Image.open(image_name_path).convert('RGB')
        img_original = numpy.array(pil_image)
        img_data = pil_to_tensor(pil_image)
        # singleton_batch = {'img_data': img_data[None].cpu()}
        singleton_batch = {'img_data': img_data[None].cuda()}
        output_size = img_data.shape[1:]


        with torch.no_grad():
            scores = segmentation_module(singleton_batch, segSize=output_size)

        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        # 获取整个分割后的结果
        vs_total = visualize_result(img_original, pred)
        # 保存分割后图像
        image = PIL.Image.fromarray(vs_total)
        image.save('{}/{}'.format(image_result_path, image_name))

        # 创建空的列表记录结果
        rate_list = ['%s'%image_name,]
        for n in range(id_save):
            #vs_single是单一要素占比图像
            vs_single = visualize_result(img_original, pred, n)
            # 计算分割后结果占比
            num = np.count_nonzero((vs_single != [0, 0, 0]).all(axis = 2))
            all_num = vs_single.shape[0] * vs_single.shape[1]
            rate = num / all_num
            rate_list.append(rate)

        # 将结果写入csv
        with open('%s' % image_result_csv ,'a' ,newline='') as f:
            writer = csv.writer(f)
            writer.writerow(rate_list)

        print(' [-] 处理完成图片：',image_name_path)
end_time = time.time()
print(' [-] 处理完成所有图片,耗时{:.2f}秒'.format(end_time - start_time))
