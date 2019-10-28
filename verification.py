import csv
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from my_image_folder import ImageFolder
import numpy as np

# transforms = transforms.Compose([
#             transforms.RandomResizedCrop(300),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

def verification(model):
    dict_content = {'FileName':'Code'}
    dataset = ImageFolder('C:\CR_\data\Test', split=1, mod='verification')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model = torch.load(model)
    model.eval()

    for i, data in enumerate(trainloader, 0):
        # torch.cuda.empty_cache()
        inputs, file_name = data
        inputs = Variable(inputs).cuda()
        outputs = model(inputs)
        outputs = torch.topk(outputs, 1)[1].squeeze(1)
        dict_content[file_name[0]] = outputs.item()
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