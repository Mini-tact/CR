"""
Author：huangwei
Time:2019.10.17
目标检测难做，通过判断小图像之间的相似性来完成判断有没有其他目标
"""
from skimage.measure import compare_ssim
from PIL import Image
from skimage import data, filters
import matplotlib.pyplot as plt
import numpy as np
import os


def pictureSimilarity(num, path):
    img = Image.open(path).convert('RGB')
    wide, high = img.size[0], img.size[1]  # 2976*3968
    img = np.asarray(img)


    img1 = img[0:int(high/2), :]
    img2 = img[int(high/2):2*int(high/2), :]

    img3 = img[:, 0:int(wide/2)]
    img4 = img[:, int(wide/2):2*int(wide/2)]

    ssim_1 = compare_ssim(img1, img2, multichannel=True)
    ssim_2 = compare_ssim(img3, img4, multichannel=True)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(321)
    plt.title('('+str(wide)+',' + str(high)+')')
    plt.imshow(img, plt.cm.gray)

    plt.annotate(str(ssim_1)+'\n'+str(ssim_2), (wide, high))  # x,y 为坐标值

    plt.subplot(323)
    plt.title('img1')
    plt.imshow(img1,plt.cm.gray)

    plt.subplot(324)
    plt.title('img2')
    plt.imshow(img2, plt.cm.gray)

    plt.subplot(325)
    plt.title('img3')
    plt.imshow(img3, plt.cm.gray)

    plt.subplot(326)
    plt.title('img4')
    plt.imshow(img4, plt.cm.gray)

    plt.savefig('C:\CR_\data\similarity\{}.png'.format(num))
    plt.close()

if __name__=="__main__":
    root = r'C:\CR_\data\Train'
    list_name = os.listdir(root)
    for num, file_name in enumerate(list_name):
        pictureSimilarity(num, os.path.join(root, file_name))