import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=13,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, 2, 1),
                                         torch.nn.BatchNorm2d(64),
                                         torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 2, 2, 0),
                                         torch.nn.BatchNorm2d(128),
                                         torch.nn.ReLU())

        self.mlp1 = torch.nn.Linear(2 * 2 * 128, 100)
        self.mlp2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x
