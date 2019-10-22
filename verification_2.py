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
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

def pictureSimilarity(array, similarity):
    """
    求解图片中是否有其他的物体
    :param array: 三维图片,(width,hight,channel_number)
    :return:图片上下切割的相似性， 图片左右切割的相似性
    """
    high, wide = array.shape[0], array.shape[1]  # 2976*3968
    img1 = array[0:int(high / 2), :]
    img2 = array[int(high / 2):2 * int(high / 2), :]

    img3 = array[:, 0:int(wide / 2)]
    img4 = array[:, int(wide / 2):2 * int(wide / 2)]

    ssim_1 = compare_ssim(img1, img2, multichannel=True)
    ssim_2 = compare_ssim(img3, img4, multichannel=True)

    """
    越小越不相似
    """
    if ssim_1 > similarity or ssim_2 > similarity:
        return True  # 相似
    else:
        return False  # 不相似

def extractionCloud(input):
    """
    1. 将图片进行判断是否含有其他的物体
    2. 使用阈值将图片进行除云以外的物体过滤，获得二值图像
    :return:
    """
    # 将数据加载类中的Tensor转换成numpy中的array并转化成(width,hight,channel_number)
    input = input.numpy()  # 形状为(mini-batch-number, channel_number, width,hight)
    input_squeeze = np.squeeze(input)  # 降维(channel_number, hight, width)
    input_squeeze = input_squeeze.reshape(input_squeeze.shape[0], input_squeeze.shape[1], 3)

    # 判断是否有其他的目标
    if pictureSimilarity(input_squeeze, 0.7) is False:
        thresh = filters.threshold_isodata(input_squeeze)  # 返回一个阈值
        input_dst = (input_squeeze <= thresh) * 1.0  # 根据阈值进行分割,得到一个二值图像
    else:
        input_dst = (input_squeeze <= 0) * 1.0

    # plt.figure('thresh', figsize=(8, 8))
    # plt.axis('off')
    # plt.subplot(121)
    # #plt.title('original image', plt.cm.gray)
    # plt.imshow(input_squeeze)
    #
    # plt.subplot(122)
    # #plt.title('binary image', plt.cm.gray)
    # plt.imshow(input_dst, plt.cm.get_cmap())
    #
    # plt.show()
    return input_squeeze, input_dst

def discardingThePictures(data_dst):
    """
    丢弃其他目标占比比较大的图片
    :return:
    """
    data_axis_sum = data_dst.sum(axis=2)  # (width, hight, channel_number)
    num = str(data_axis_sum.tolist()).count("0.0")  # 计算矩阵中0的个数
    if np.true_divide(num, 900) > 0.6:  # 这个阈值需要调整
        return True  # 不需要舍弃
    return False  # 舍弃图片


def cutPictureAndComputResult(input, model, windows_size = 300):
    """
    1. 将图片进行判断是否含有其他的物体
    2. 使用阈值将图片进行除云以外的物体过滤，获得只含云的图片
    3. 将图片按照300*300的窗口进行裁剪，将不含云的图片区域按照策略进行舍去
    4. 将切割后的图片进行预测
    5. 对多个预测结果进行累加取最大值作为未切割图片的预测值
    :param input:
    :return:
    """
    # 多个图片训练结果的集合
    result_dict = {}

    # 加载模型、并进入测试
    model = torch.load(model)
    model.eval()
    # 判断是否含有其他的物体
    img, img_dst = extractionCloud(input)
    # 切割
    high, wid =img.shape[0], img.shape[1]
    # hig_num = int(high/windows_size)
    # wid_num = int(wid/windows_size)
    # for hig_ in range(hig_num):
    #     for wid_ in range(wid_num):
    #         # 判断是否可以舍弃切割后的图像
    #         img_new = img[hig_*windows_size:(hig_+1)*windows_size, wid_*windows_size:(wid_+1)*windows_size]
    #         img_new_dst = img_dst[hig_ * windows_size:(hig_ + 1) * windows_size, wid_ * windows_size:(wid_ + 1) * windows_size]
    hig_num_num = int((high - windows_size) / 150)
    wid_num_num = int((wid - windows_size) / 150)


    a = 0
    for hig_ in range(hig_num_num):
        for wid_ in range(wid_num_num):
            img_new = img[hig_ * int((windows_size/2)):(hig_ + 2) * int((windows_size/2)), wid_ * int((windows_size/2)):(wid_ + 2) * int((windows_size/2))]
            img_new_dst = img_dst[hig_ * int((windows_size/2)):(hig_ + 2) * int((windows_size/2)), wid_ * int((windows_size/2)):(wid_ + 2) * int((windows_size/2))]

            a += 1
            if hig_num_num != 0 and wid_num_num != 0:
                plt.subplot(str(hig_num_num)+str(wid_num_num)+str(a))
                plt.axis('off')
                plt.imshow(img_new)

            if discardingThePictures(img_new_dst) is False:
                continue  #  占比大 舍弃图片不训练

            # 规整数据，使得数据符合训练的数据格式
            #img_new = img_new.reshape(3, img_new.shape[0], img_new.shape[1])
            # 转成Tensor格式，并标准化
            inputs = Variable(torch.unsqueeze(transforms(img_new), 0))

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            # 训练并得到结果
            outputs = model(inputs)
            outputs = torch.topk(outputs, 1)[1].squeeze(1).item()

            if str(outputs) not in result_dict.keys():
                result_dict[outputs] = 1
            else:
                result_dict[outputs] = int(result_dict[outputs]) + 1
    plt.show()
    print(result_dict)
    if result_dict:
        return int(max(zip(result_dict.values(), result_dict.keys()))[0])  # 返回字典中最大值对应的标签
    else:
        return 1





def verification(model):
    dict_content = {'FileName': 'Code'}
    dataset = ImageFolder('C:\CR_\data\Test', split=1, mod='verification')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for i, data in enumerate(trainloader, 0):
        # 取出一张图片
        inputs, file_name = data
        """
        1. 将图片进行判断是否含有其他的物体
        2. 使用阈值将图片进行除云以外的物体过滤，获得只含云的图片
        3. 将图片按照300*300的窗口进行裁剪，将不含云的图片区域按照策略进行舍去
        4. 将切割后的图片进行预测
        5. 对多个预测结果进行累加取最大值作为未切割图片的预测值
        """
        dict_content[file_name[0]] = cutPictureAndComputResult(inputs, model)
    print(dict_content)

    """
    写入文件
    """
    csvFile3 = open('C:\\CR_\\Train_label.csv', 'w', newline='')
    writer2 = csv.writer(csvFile3)
    for key in dict_content.keys():
        writer2.writerow([key, dict_content[key]])
    csvFile3.close()

    """
    文件处理
    复制到备份文件夹、新建Train_label.csv  
    """

if __name__ == "__main__":
    verification('C:\\CR_\\model\\net_googlenet.pkl')