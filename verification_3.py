import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import csv
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from my_image_folder import ImageFolder
from skimage import data, filters
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import torchvision.transforms as transforms

transforms = transforms.Compose([
            transforms.ToTensor()])

def resize_by_width(cls, w_divide_h):
    """按照宽度进行所需比例缩放"""
    im = Image.open(cls)
    (x, y) = im.size
    if x < y:
        im.rotate(90)
        (x, y) = (y, x)
    y_s = int(w_divide_h*y/x)
    return np.array(im.resize((w_divide_h, y_s), Image.ANTIALIAS))

def getCloud(arr):
    """
    给出一个索引来找到正确的云
    方法：
    :param arr:
    :return:
    """
    # row_ave.index(max(row_ave))
    row_row_ave = []
    for item in range(0, len(arr) - 250):
        row_row_ave.append(sum(arr[item:item+250]))
    if len(row_row_ave) == 0:
        return 125
    return row_row_ave.index(max(row_row_ave)) + 125



if __name__=='__main__':
    dict_content = {'FileName': 'Code'}
    root = r'C:\CR_\data\Test'
    save_path = 'C:\CR_\data\Test'
    # 加载模型、并进入测试
    model = torch.load(r'C:\CR_\model\net_ieee_10_28.pkl')
    model.eval()
    for picture in os.listdir(root):
        # 按比例缩放，缩放成 500*250
        np_image = resize_by_width(root + '\\' + picture, 500)
        # 找到图像中的云所在的区域，并切割
        row_ave = []
        for row in np_image:
            # row[row > 250] = 125  # 去掉太阳
            row_ave.append(np.mean(row))
        row_num = getCloud(row_ave)
        # 防止在图片嘴上或者最下的地方
        if row_num <= 125:
            image = np_image[0:250, :]
        elif row_num >= len(row_ave)-125:
            image = np_image[len(row_ave)-125:, :]
        else:
            image = np_image[row_num-125:row_num+125, :]

        # 训练
        image = transforms(image)
        image = image.unsqueeze(0)
        outputs = model(image.cuda()).cpu()
        dict_content[picture] = torch.topk(outputs, 1)[1].squeeze(1).item()

    print(dict_content)
    """
    写入文件
    """
    csvFile3 = open('C:\\CR_\\Train_label.csv', 'w', newline='')
    writer2 = csv.writer(csvFile3)
    for key in dict_content.keys():
        writer2.writerow([key, dict_content[key]])
    csvFile3.close()
