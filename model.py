import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # self.cam = ChannelAttentionModule(13,3)
        # self.sam = SpatialAttentionModule()
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

    def forward(self, x):
        # catt = self.cam(x)
        # x = catt*x
        # satt = self.sam(x)
        # x = satt*x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.dropout(x)
        x = F.relu(x)
        x = self.mlp2(x)
        # return x,catt,satt
        return x
    
    def __repr__(self):
        return "SimpleCNNClassification"


class SimpleCNNRegression(nn.Module):
    def __init__(self):
        super(SimpleCNNRegression, self).__init__()
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

        self.mlp1 = torch.nn.Linear(2 * 2 * 128, 64)
        self.dropout = nn.Dropout(0.5)
        self.mlp21 = torch.nn.Linear(64, 2)
        self.mlp22 = torch.nn.Linear(64, 2)
        self.mlp3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.dropout(x)
        x1 = self.mlp21(x)
        x2 = self.mlp22(x)
        x = F.relu(self.mlp3(torch.cat((x1,x2),1)))
        x1 = F.softmax(x1,1)
        return x1,x.squeeze(1)
    def __repr__(self):
        return "SimpleCNNRegression"

class SimpleCNNTwoTask(nn.Module):
    def __init__(self):
        super(SimpleCNNTwoTask, self).__init__()
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
        self.mlp2_1 = torch.nn.Linear(100, 2)
        self.mlp2_2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.dropout(x)
        y = self.mlp2_1(x)
        z = self.mlp2_2(x)
        return y, F.relu(z)
    def __repr__(self):
        return "SimpleCNNTwoTask"

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionTwoTask(nn.Module):
    def __init__(self):
        super(InceptionTwoTask, self).__init__()
        self.conv1 = InceptionA(13,pool_features=32)
        self.conv2 = InceptionA(256,pool_features=64)
        self.conv3 = InceptionA(288,pool_features=64)
        self.conv4 = InceptionB(288)
        self.conv51 = InceptionC(768, channels_7x7=128)
        self.conv52 = InceptionC(768, channels_7x7=160)
        self.conv53 = InceptionC(768, channels_7x7=160)
        self.conv54 = InceptionC(768, channels_7x7=192)
        self.pool = nn.MaxPool2d(2)
        
        self.mlp = torch.nn.Linear(768 * 3 * 3, 100)
        self.dropout = nn.Dropout(0.5)
        self.mlp2_1 = torch.nn.Linear(100, 2)
        self.mlp2_2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)
        x = self.pool(x)
        x = x.view(-1,768 * 3 * 3)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.mlp2_2(x)
        return x.squeeze(1)

    def __repr__(self):
        return "SimpleCNNTwoTask"