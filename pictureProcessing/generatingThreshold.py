from PIL import Image
from skimage import data, filters
import matplotlib.pyplot as plt
import numpy as np

# 读取一张图片转化成numpy类型
path = "C:\\CR_\\data\\2_05c4e98642874de3a1be80218436bc8c.png"
path_1 = r'C:\CR_\data\0cd626a4b2144bf0bb05e5d114eb85dc.png'
path_2 = r'C:\CR_\data\1_0f1c21350599491c89ebc8eef45e272e.jpg'
path_3 = r'C:\CR_\data\1_fae5861fa2234f9b8ed250cb07c9ca4f.jpg'
path_4 = r'C:\CR_\data\2_0f1481b2a05e453390f6d104e7b9af68.jpg'
image = Image.open(path_3).convert('L')
image = np.asarray(image)

# 目标检测图中含有几类物体


thresh = filters.threshold_isodata(image)   #返回一个阈值
dst =(image <= thresh)*1.0   #根据阈值进行分割

plt.figure('thresh',figsize=(8, 8))

plt.subplot(121)
plt.title('original image')
plt.imshow(image,plt.cm.gray)

plt.subplot(122)
plt.title('binary image')
plt.imshow(dst, plt.cm.gray)

plt.show()