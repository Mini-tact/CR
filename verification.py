import csv

import torch
from torch.autograd import Variable
import xlrd
from my_image_folder import ImageFolder

def verification():
    dict_content = {'FileName':'Code'}
    dataset = ImageFolder('C:\CR_\data\Test', split=1, mod='verification')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model = torch.load('C:\\CR_\\net.pkl')
    model.eval()

    for i, data in enumerate(trainloader, 0):
        # torch.cuda.empty_cache()
        if torch.cuda.is_available():
            inputs, file_name = data
            inputs= Variable(inputs).cuda()
        else:
            inputs = Variable(data)
        outputs = model(inputs)
        outputs = torch.topk(outputs, 1)[1].squeeze(1)
        dict_content[file_name[0]] = outputs.item()
    print(dict_content)

    """
    写入文件
    """
    csvFile3 = open('C:\\CR_\\verification.csv', 'w', newline='')
    writer2 = csv.writer(csvFile3)
    for key in dict_content.keys():
        writer2.writerow([key, dict_content[key]])
    csvFile3.close()

if __name__ == "__main__":
    verification()