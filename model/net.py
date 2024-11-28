import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(512, 35)
        self.fc3 = nn.Linear(35, 10)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(self.bn1(F.max_pool2d(self.conv2(x), 2)))
        # x = self.dropout1(x)
        x = F.relu(self.conv3(x), 2)
        x = F.relu(self.bn2(F.max_pool2d(self.conv4(x), 2)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)