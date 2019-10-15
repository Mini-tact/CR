import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dirt, rate):
    """
    获取文件下的所有文件名
    :param dirt:
    :return:
    """
    images = []
    if not os.path.isdir(dirt):
        return None
    for root, _, fnames in sorted(os.walk(dirt)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, fname)
                images.append(item)
    shuffle(images)
    return images[0:int(len(images) * rate)], images[int(len(images) * rate):]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, root, loader=default_loader, split=0.8, mod='train'):
        self.imgs_train, self.imgs_test = make_dataset(root, split)
        self.loader = loader
        self.mod = mod
        self.transform = transforms.Compose([
            # ValueError: empty range for randrange() (0,-59, -59)
            # http://www.manongzj.com/blog/3-ijvagxsghnolbvx.html 文件大小太小超出切割范围
            transforms.RandomCrop((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        if self.mod == 'train' or 'verification':
            path, target = self.imgs_train[index]
        else:
            path, target = self.imgs_test[index]
        img = self.loader(path)
        print(path)
        image = self.transform(img)

        # 转成one_hot编码标签
        # target = torch.tensor(np.array(), dtype=torch.int64)
        # target_tensor = torch.LongTensor([[float(target.split('_')[0])]])  # 注意是[[]]
        # target_hot = torch.zeros(1, 29).scatter_(1, target_tensor, 1)
        if self.mod == 'train' or self.mod == 'test':
            target = float(target.split('_')[0])
            return image, target
        if self.mod == 'verification':
            return image, target


    def __len__(self):
        if self.mod == 'train' or self.mod == 'verification':
            return len(self.imgs_train)
        else:
            return len(self.imgs_test)
