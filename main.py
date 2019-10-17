# -*- coding: utf-8 -*-
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np

from model.net_ieee import Net
from my_image_folder import ImageFolder
from verification import verification
import torchvision.models as models

loss_function = nn.MSELoss()
function_loss_L1 = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

def t_training(testset, net):
    with torch.no_grad():
        # torch.backends.cudnn.enabled = False
        loader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=3, pin_memory=True)#  pin_memory 将数据从RAM直接转移到GPU

        all_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = labels.squeeze()
            outputs = net(inputs.cuda())
            all_loss = all_loss + function_loss_L1(outputs.cpu(), labels.long())
    return all_loss/i

if __name__ == "__main__":
    start_time = time.clock()
    losses_his = [[], []]
    train_acc = 0
    # 数据集加载
    trainset = ImageFolder('C:\CR_\data\Train', split=0.8, mod='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=6, pin_memory=True)
    testset = ImageFolder('C:\CR_\data\Train', split=0.8, mod='test')
    # 模型加载
    #net = Net()
    net = models.googlenet() #pretrained=True
    # 使用CUDA训练模型
    if torch.cuda.is_available():
        net.cuda()
    # 初始化权重
    # nn.init.xavier_uniform_(net.conv1[0].weight.data, gain=1)
    # nn.init.constant_(net.conv1[0].bias.data, 0.1)
    # nn.init.xavier_uniform_(net.conv2[0].weight.data, gain=1)
    # nn.init.constant_(net.conv2[0].bias.data, 0.1)
    # nn.init.xavier_uniform_(net.conv3[0].weight.data, gain=1)
    # nn.init.constant_(net.conv3[0].bias.data, 0.1)
    # nn.init.xavier_uniform_(net.conv4[0].weight.data, gain=1)
    # nn.init.constant_(net.conv4[0].bias.data, 0.1)
    # 是指优化策略
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 0.0001

    # 进入训练阶段
    for epoch in range(120):
        train_loss = 0.0
        all_loss = 0.0

        net.train()
        # mini-batch
        for i, data in enumerate(trainloader, 0):
            # torch.cuda.empty_cache()
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # labels = labels.reshape(-1, 1)
            # labels = torch.zeros(200, 29).scatter_(1, labels.long(), 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = function_loss_L1(outputs.logits, labels.long())  # .logits
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # train_acc += torch.sum(outputs == labels.data)

        all_loss = t_training(testset, net.eval())
        print('epoch{}, Train Loss: {:.6f}, Test Loss:{:.6f}'.format(epoch+1, train_loss / i, all_loss))
        losses_his[0].append(train_loss / i)
        losses_his[1].append(all_loss)

    torch.save(net, 'C:\\CR_\\model\\net_googlenet.pkl')

    """
    绘制误差图像
    """
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel("miniBatch的个数")
    plt.ylabel("每个miniBatch的误差")
    l1, = plt.plot(np.linspace(0, len(losses_his[0]), len(losses_his[0])), losses_his[0])
    l2, = plt.plot(np.linspace(0, len(losses_his[1]), len(losses_his[1])), losses_his[1], linestyle='--')
    plt.legend(handles=[l1, l2], labels=['训练集误差', '测试集误差'], loc='best')
    plt.show()
    plt.figure()
    plt.plot(losses_his[0], losses_his[1])
    plt.show()

    # 验证集
    verification('C:\\CR_\\model\\net_googlenet.pkl')
    print('模型训练时间为{}'.format(time.clock() - start_time))

