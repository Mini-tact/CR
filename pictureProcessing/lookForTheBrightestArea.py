import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

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
    root = r'C:\CR_\data\Test'
    save_path = 'C:\CR_\data\Test'
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


        #Image.fromarray(np.uint8(image)).save(os.path.join(save_path, picture))
