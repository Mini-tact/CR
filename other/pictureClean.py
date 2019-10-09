"""
@huangwei 2019.10.9
删除图片中通道数不是3的图片
"""
import os
import numpy as np
from PIL import Image

for picture in os.listdir('C:\CR\data\Train'):
    image = Image.open('C:\CR\data\Train'+'\\'+picture)
    np_image = np.array(image)
    if np_image.shape[2] !=3:
        os.remove(os.path.join('C:\CR\data\Train', picture))
        print("Delete File: " + os.path.join('C:\CR\data\Train', picture))

