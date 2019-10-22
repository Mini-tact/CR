import math

from PIL import Image
import numpy as np
import os
from skimage import transform, data
def resizePicture(path, window_size):
    """
    重新改变图片的大小
    :param path:源图片的路径
    :return:处理后的图片、图片的宽、图片的高
    """
    img = Image.open(path).convert('RGB')
    wid, hig = img.size[0], img.size[1]  # 宽、高
    img = np.asarray(img)
    if wid < window_size:
        img = transform.rescale(img, [math.ceil(window_size / wid), 1])
    if hig < window_size:
        img = transform.rescale(img, [1, math.ceil(window_size / wid)])

    return img, img.shape[0], img.shape[1]

def cutPicture(path, windows_size=300):
    """
    将图片按照窗口的大小进行切分
    :param path:
    :param windows_size:
    :return:
    """
    # 对图片的大小进行resize
    img, wide, high =resizePicture (path, windows_size)
    hig_num = int(wide/windows_size)
    wid_num = int(high/windows_size)

    """
    图片切割
    """
    for wid_ in range(wid_num):
        for len_ in range(hig_num):
            img_new = img[len_*windows_size:(len_+1)*windows_size, wid_*windows_size:(wid_+1)*windows_size]
            image = Image.fromarray(np.uint8(img_new))

            path = path.replace('C:\CR_\data\Train','C:\CR_\data\cut_picture')
            path_new = path.split('.')[0]+'_'+str(wid_)+'_'+str(len_)+'.png'
            image.save(path_new)

if __name__=="__main__":
    root = r'C:\CR_\data\Train'
    list_name = os.listdir(root)
    for num, file_name in enumerate(list_name):
        print('正在处理图片{}'.format(file_name))
        cutPicture(os.path.join(root, file_name))


    # transform.rescale(img, 0.1)
    # skimage.transform.resize(image, [300,300])