"""
Author：huangwei
Time:2019.10.10
将文件名称带有‘;’的文件移动到data/else文件夹下
"""
import os
import shutil

list_name = os.listdir('C:\\CR\\data\\Train')

for item in list_name:
    if len(item.split(';')) == 2:
        shutil.move(os.path.join('C:\CR\data\Train', item), os.path.join('C:\CR\data\else', item))
        print('remove {} into data\\else'.format(os.path.join('C:\CR\data\Train', item)))