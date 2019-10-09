# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
from my_image_folder import ImageFolder
from net_ieee import Net

loss_function = nn.MSELoss()
function_loss_L1 = nn.L1Loss()

def t_training(testset, net):
    with torch.no_grad():
        # torch.backends.cudnn.enabled = False
        loader = torch.utils.data.DataLoader(testset, batch_size=50, num_workers=3)

        all_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            outputs = net(inputs)
            all_loss = all_loss + function_loss_L1(outputs.cpu()[:, 0], labels.float())
    return all_loss/i

if __name__ == "__main__":
    losses_his = [[], []]
    train_acc = 0
    # 数据集加载
    trainset = ImageFolder('C:\CR\data\Train', split=0.8, mod='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=3)
    testset = ImageFolder('C:\CR\data\Train', split=0.8, mod='test')
    # 模型加载
    net = Net()
    # 使用CUDA训练模型
    if torch.cuda.is_available():
        net.cuda()
    # 初始化权重
    nn.init.xavier_uniform_(net.conv1[0].weight.data, gain=1)
    nn.init.constant_(net.conv1[0].bias.data, 0.1)
    nn.init.xavier_uniform_(net.conv2[0].weight.data, gain=1)
    nn.init.constant_(net.conv2[0].bias.data, 0.1)
    nn.init.xavier_uniform_(net.conv3[0].weight.data, gain=1)
    nn.init.constant_(net.conv3[0].bias.data, 0.1)
    nn.init.xavier_uniform_(net.conv4[0].weight.data, gain=1)
    nn.init.constant_(net.conv4[0].bias.data, 0.1)
    # 是指优化策略
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 0.0001

    # 进入训练阶段
    for epoch in range(100):
        train_loss = 0.0
        all_loss = 0.0

        net.train()
        # mini-batch
        for i, data in enumerate(trainloader, 0):
            # torch.cuda.empty_cache()
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            labels = labels.reshape(-1, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = function_loss_L1(outputs, labels.float())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # train_acc += torch.sum(outputs == labels.data)

        all_loss = t_training(testset, net.eval())
        print('epoch{}, Train Loss: {:.6f}, Test Loss:{:.6f}'.format(epoch+1, train_loss / i, all_loss))
        losses_his[0].append(train_loss / i)
        losses_his[1].append(all_loss)

    torch.save(net, 'C:\\CR\\net.pkl')
