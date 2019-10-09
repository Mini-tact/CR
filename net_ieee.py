import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=400, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(400),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=400, out_channels=576, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(576),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=1024, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        # self.fc1 = nn.Linear(12544, 3136)
        # self.fc2 = nn.Linear(3136, 392)
        # self.fc3 = nn.Linear(392, 1)
        self.fc1 = nn.Linear(14400, 1440)
        self.fc2 = nn.Linear(1440, 144)
        self.fc3 = nn.Linear(144, 1)


    def forward(self, x):
        x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
