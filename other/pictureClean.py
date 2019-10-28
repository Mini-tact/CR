"""
@huangwei 2019.10.9
删除图片中通道数不是3的图片
"""
import os
import numpy as np
from PIL import Image

#for picture in os.listdir('C:\CR\data\Train'):
    # image = Image.open('C:\CR\data\Train'+'\\'+picture)
    # np_image = np.array(image)
    # if np_image.shape[2] !=3:
    #     os.remove(os.path.join('C:\CR\data\Train', picture))
    #     print("Delete File: " + os.path.join('C:\CR\data\Train', picture))
    #
    #

# 删除尺寸不对的图片
sum_ = 0
for picture in os.listdir(r'C:\CR_\data\img'):
    im = Image.open(os.path.join('C:\CR_\data\img', picture))
    x, y = im.size
    if x == 500 and y == 250:
        continue
    sum_ += 1
    print('不符合尺寸{}'.format(sum_, picture))
    im.close()
    os.remove('C:\CR_\data\img'+'\\'+ picture)