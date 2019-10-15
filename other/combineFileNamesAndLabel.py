"""
huangwei 2019.10.8
将数据集的标签作为图片的名称，方便读取
1.读取图片
2.读取标签
3.将图片和标签匹配
"""
import os
import pandas as pd  # 导入pandas包
import numpy as np
if __name__ == "__main__":
    # 读取图片
    training_path ="C:\\CR\\data\\Train"
    test_path = "C:\\CR\\data\\Test"
    listdir = os.listdir(test_path)
    # 读取标签
    label_path = 'C:\\CR\\data\\Train_label.csv'
    data = pd.read_csv(label_path)  # 读取csv文件
    FileName = data.loc[:, ['FileName']].values.reshape(-1)
    Code = data.loc[:, ['Code']].values.reshape(-1)

    for picture in listdir:
        if picture in FileName:
            number = np.where(FileName == picture)[0][0]
            file_path = 'C:\\CR\\data\\Train'
            # 原来的名称
            raw_name = file_path + '\\' + picture
            # 新的名称
            new_name = file_path + '\\' + Code[number] + '_' + picture
            # 改名称
            # os.rename(raw_name, new_name)
        else:
            print('标签不存在')


#  end