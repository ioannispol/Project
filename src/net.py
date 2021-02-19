import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.pool3 = nn.MaxPool2d(2, 2)

        # input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(64*23*23, self.num_classes)

    # Function to calculate the input dimension to Linear layer
    # TODO: Implement calculation depending on the network structure
    # def calc_input_dims(self):
    #     batch_data = torch.zeros((1, 3, 224, 224))
    #     batch_data = self.conv1(batch_data)
    #     batch_data = self.pool1(batch_data)
    #     batch_data = self.conv2(batch_data)
    #     batch_data = self.conv3(batch_data)
    #     batch_data = self.pool2(batch_data)
    #     batch_data = self.conv4(batch_data)
    #     batch_data = self.pool3(batch_data)

    #     return int(np.prod(batch_data.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool3(x)
        # print(x.shape)
        x = x.view(-1, 64 * 23 * 23)
        x = self.fc1(x)

        return x
