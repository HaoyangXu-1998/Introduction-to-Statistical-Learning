import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(13*15*15, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 2),
        ) 
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(x),None,None
    
    def __repr__(self):
        return "FFN"

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class SimpleCNNClassification(nn.Module):
    def __init__(self):
        super(SimpleCNNClassification, self).__init__()
        self.cam = ChannelAttentionModule(13,3)
        self.sam = SpatialAttentionModule()
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
        self.dropout = nn.Dropout(0.5)
        self.mlp2 = torch.nn.Linear(100, 2)
        nn.init.kaiming_normal_(self.mlp1.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.mlp2.weight, mode="fan_in")

    def forward(self, x):
        catt = self.cam(x)
        x = catt*x
        satt = self.sam(x)
        x = satt*x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.mlp2(x)
        return x,catt,satt
    
    def __repr__(self):
        return "SimpleCNNClassification"


class SimpleCNNRegression(nn.Module):
    def __init__(self):
        super(SimpleCNNRegression, self).__init__()
        self.cam = ChannelAttentionModule(13,3)
        self.sam = SpatialAttentionModule()
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
        self.dropout = nn.Dropout(0.5)
        self.mlp2 = torch.nn.Linear(100, 1)
        nn.init.kaiming_normal_(self.mlp1.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.mlp2.weight, mode="fan_in")

    def forward(self, x):
        catt = self.cam(x)
        x = catt*x
        satt = self.sam(x)
        x = satt*x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        x = F.relu(x)
        return x.squeeze(1),catt,satt
    
    def __repr__(self):
        return "SimpleCNNRegression"