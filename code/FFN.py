# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import time
# import matplotlib.pyplot as plt
# %%
BATCH_SIZE=512
EPOCHS=600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
Y = []
orbit_list = []
for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    dir_month = os.path.join("/tank/utsumi/PMM/MATCH.GMI.V05A/S1.ABp083-137.DPRGMI.V06A.9ave.surfPrecipTotRate/2017", month)
    days = os.listdir(dir_month)
    days.sort()
    for day in days:
        dir_day = os.path.join(dir_month, day)
        files = os.listdir(dir_day)
        files.sort()
        for file in files:
            tmp = np.load(os.path.join(dir_day, file))
            tmp = tmp[7:2955:15, 27]
            orbit_list.append(str.split(file, '.')[1])
            Y.append(tmp)
Y = np.concatenate(Y)
# %%
X = []
for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    dir_month = os.path.join("/tank/utsumi/PMM/MATCH.GMI.V05A/S1.ABp103-117.GMI.Tc/2017", month)
    days = os.listdir(dir_month)
    days.sort()
    for day in days:
        dir_day = os.path.join(dir_month, day)
        files = os.listdir(dir_day)
        files.sort()
        for file in files:
            if str.split(file, '.')[1] in orbit_list:
                tmp = np.load(os.path.join(dir_day, file))
                tmp = tmp[:2955].reshape(-1, 15, 15, 9)
                X.append(tmp)
            else:
                continue
X = np.concatenate(X)
# X = np.rollaxis(X, 3, 1)
# %%
X2 = []
for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    dir_month = os.path.join("/tank/utsumi/PMM/MATCH.GMI.V05A/S1.ABp103-117.GMI.TcS2/2017", month)
    days = os.listdir(dir_month)
    days.sort()
    for day in days:
        dir_day = os.path.join(dir_month, day)
        files = os.listdir(dir_day)
        files.sort()
        for file in files:
            if str.split(file, '.')[2] in orbit_list:
                tmp = np.load(os.path.join(dir_day, file))
                tmp = tmp[:2955].reshape(-1, 15, 15, 4)
                X2.append(tmp)
            else:
                continue
X2 = np.concatenate(X2)
# X2 = np.rollaxis(X2, 3, 1)
# %%
X = np.concatenate((X, X2), axis=3)
# %%
np.save('./data/X.npy', X)
np.save('./data/Y.npy', Y)



# %%
Land = []
for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    dir_month = os.path.join("/tank/utsumi/PMM/MATCH.GMI.V05A/S1.ABp103-117.GMI.surfaceTypeIndex/2017", month)
    days = os.listdir(dir_month)
    days.sort()
    for day in days:
        dir_day = os.path.join(dir_month, day)
        files = os.listdir(dir_day)
        files.sort()
        for file in files:
            if str.split(file, '.')[1] in orbit_list:
                tmp = np.load(os.path.join(dir_day, file))
                tmp = tmp[7:2955:15, 7]
                Land.append(tmp)
            else:
                continue
Land = np.concatenate(Land)
Land = Land[mask_missing]
# %%
Y = np.load('../data/Y.npy')
X = np.load('../data/X.npy')
Land = np.load('../data/Land.npy')
mask_missing = Y>=0
Y = Y[mask_missing]
X = X[mask_missing]
Land = Land[mask_missing]
# %%
for i in range(20):
    np.save("../data/Y_%s.npy" % str(i), Y[int(idx[i]):int(idx[i+1])])
# %%
Y[Y>0] = 1
# %%
# Ocean specific
mask_ocean = Land==1
X_ocean = X[mask_ocean]
Y_ocean = Y[mask_ocean]

# %%
Land[(Land>=4) & (Land<=7)] = 3 # Vegetation
Land[(Land>=8) & (Land<=11)] = 4 # Snow
Land[Land==12] = 5 # Standing water
Land[Land==13] = 6 # Land/ocean or water Coast
Land[Land==14] = 7 # Sea-ice edge

# %%
np.random.seed(999)
idx = np.random.choice(len(Y_ocean), len(Y_ocean), False)

# %%
# central
train_len = 423424
val_len = 141269
X_train = X_ocean[idx[:train_len], 6:9, 6:9, :]
X_val   = X_ocean[idx[train_len:(train_len+val_len)], 6:9, 6:9, :]
X_test  = X_ocean[idx[(train_len+val_len):], 6:9, 6:9, :]
Y_train = Y_ocean[idx[:train_len]]
Y_val   = Y_ocean[idx[train_len:(train_len+val_len)]]
Y_test  = Y_ocean[idx[(train_len+val_len):]]
# %%
del X, Y, Land, mask_missing, mask_ocean

# %%
X_train = X_train[:, 1, 1, :]
X_val   = X_val[:, 1, 1, :]
X_test  = X_test[:, 1, 1, :]
# %%
tmp = np.unique(np.where(X_train < 0)[0])
tmp = np.setdiff1d(np.arange(len(X_train)), tmp)
X_train = X_train[tmp]
Y_train = Y_train[tmp]
tmp = np.unique(np.where(X_val < 0)[0])
tmp = np.setdiff1d(np.arange(len(X_val)), tmp)
X_val = X_val[tmp]
Y_val = Y_val[tmp]
tmp = np.unique(np.where(X_test < 0)[0])
tmp = np.setdiff1d(np.arange(len(X_test)), tmp)
X_test = X_test[tmp]
Y_test = Y_test[tmp]
# %%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), 
batch_size=BATCH_SIZE, shuffle=True)

val_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)), 
batch_size=BATCH_SIZE, shuffle=True)

test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)), 
batch_size=BATCH_SIZE, shuffle=True)

# %%
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
# %%
class Feed_ForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.hidden = nn.Sequential(
            nn.Linear(13, 256),
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
            ############################
            nn.Dropout(0.5),
            ############################
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        ) 
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(self.bn(x))
# %%
class ReducedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.hidden = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(self.bn(x))
class ReducedNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.hidden = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(self.bn(x))

# %%
class ReducedNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.hidden = nn.Sequential(
            nn.Linear(13, 75),
            nn.ReLU(),
            nn.Linear(75, 39),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(39, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(self.bn(x))

# %%
class ReducedNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.hidden = nn.Sequential(
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            nn.ReLU(),
            nn.Linear(13, 13),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(13, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(self.bn(x))
# %%
class ReducedNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.hidden = nn.Sequential(
            nn.Linear(13, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(30, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        in_size = x.size(0) #batch_size
        x = x.view(in_size, -1)
        return self.hidden(self.bn(x))
# %%
class FocalLoss(nn.Module):
    def __init__(self,classes,alpha=None,gamma=2,size_average=False):
        super(FocalLoss,self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(classes,1))
        else:
            if isinstance(alpha,Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.classes = classes
        self.size_average = size_average

    def forward(self,inputs,targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        mask = inputs.data.new(N,C).fill_(0)
        mask = Variable(mask)
        ids = targets.view(-1,1)
        mask.scatter_(1,ids.long(),1.)
        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]
        alpha = self.alpha[ids.view(-1).long()]
        probs = (P*mask).sum(1).view(-1,1)
        logp = probs.log()
        batchloss = -alpha*(torch.pow(1-probs,self.gamma))*logp
        if self.size_average:
            loss = batchloss.mean()
        else:
            loss = batchloss.sum()
        return loss

# %%
model = ReducedNet5().to(DEVICE)
for m in model.hidden:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in")
# %%
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
#optimizer = optim.Adadelta(model.parameters(), 1e-3)
loss_function = FocalLoss(2, alpha=torch.tensor([1, 1]).to(DEVICE), 
#torch.tensor([0.25, 0.75]).to(DEVICE), 
size_average=False)
# %%
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    Loss_train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        #######################
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        #######################
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()/BATCH_SIZE))
            Loss_train.append(loss.item()/BATCH_SIZE)
    return np.array(Loss_train).mean()

# %%
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


# %%
###############################################################
########################### TRAIN #############################
###############################################################
Loss_train = []
Loss_val  = []
for epoch in range(1, EPOCHS + 1):
    start_time = time.perf_counter()
    Loss_train.append(train(model, DEVICE, train_loader, optimizer, epoch))
    Loss_val.append(test(model, DEVICE, val_loader))
    # print(time.perf_counter() - start_time)

# %%
Loss_train = np.array(Loss_train)
Loss_val  = np.array(Loss_val)
# %%
def confusionMatrix(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    PP = PN = NP = NN = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            pred_rain = pred==1
            target_rain = target.view_as(pred)==1
            PP += ( target_rain *  pred_rain).sum().item()
            PN += ( target_rain * ~pred_rain).sum().item()
            NP += (~target_rain *  pred_rain).sum().item()
            NN += (~target_rain * ~pred_rain).sum().item()
            test_loss += loss_function(output, target).item() # 将一批的损失相加
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return np.array([[PP, NP], [PN, NN]])


# %%
result = confusionMatrix(model, DEVICE, test_loader)
print(result)
# %%
# precision = result[0, 0] / (result[0, 0] + result[0, 1])
POD_ffn = result[0, 0] / (result[0, 0] + result[1, 0])
# recall    = result[0, 0] / (result[0, 0] + result[1, 0])
FAR_ffn = 1 - result[0, 0] / (result[0, 0] + result[0, 1])
#F1 = 2 * precision * recall / (precision + recall)
print("POD = ", POD_ffn, "\n FAR = ", FAR_ffn)

# %%
plt.figure(figsize=(12, 8))
plt.plot(Loss_train[1:], color="blue", label="Train_loss")
plt.plot(Loss_val[1:],   color="red",  label="Validation_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../data/Loss/Loss_30.pdf")

# %%
# ##############################
# How to save and load models
torch.save({'EPOCH': EPOCHS, 
"Feed-ForwardNet_state_dict": model.state_dict(), 
"Adam_state_dict": optimizer.state_dict(), 
"result": result, 
"POD": POD_ffn, 
"FAR": FAR_ffn,
"Loss_train": Loss_train, 
"Loss_val": Loss_val, 
"loss_func.alpha": loss_function.alpha,
"loss_func.gamma": loss_function.gamma,
"model_type": "Reduced5"
}, "/data/haoyang/MODELS/Neural_network/ReducedNet5.tar")

# %%
model = ReducedNet5()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
checkpoint = torch.load("/data/haoyang/MODELS/Neural_network/ReducedNet5.tar")
model.load_state_dict(checkpoint['Feed-ForwardNet_state_dict'])
optimizer.load_state_dict(checkpoint['Adam_state_dict'])
EPOCHS = checkpoint['EPOCH']
Loss_train = checkpoint['Loss_train']
model.to(DEVICE)
model.eval()
print(get_parameter_number(model), EPOCHS, checkpoint['result'], checkpoint['POD'], checkpoint['FAR'])
# %%
model.state_dict()['hidden.0.weight'][0]

# %%
# Try Reduced network
# ##############################
model = ReducedNet().to(DEVICE)
for m in model.hidden:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in")
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
#optimizer = optim.Adadelta(model.parameters(), 1e-3)
loss_function = FocalLoss(2, alpha=torch.tensor([1, 1]).to(DEVICE), 
#torch.tensor([0.25, 0.75]).to(DEVICE), 
size_average=False)
# %%
# Try Reduced network
# ##############################
model = ReducedNet5().to(DEVICE)
for m in model.hidden:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in")
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
#optimizer = optim.Adadelta(model.parameters(), 1e-3)
loss_function = FocalLoss(2, alpha=torch.tensor([1, 1]).to(DEVICE), 
#torch.tensor([0.25, 0.75]).to(DEVICE), 
size_average=False)


# %%
np.random.seed(123)
idx_1e5 = np.random.choice(a=len(X_train), size=100000, replace=False)
# %%
train_loader_1e5 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_train[idx_1e5]), torch.from_numpy(Y_train[idx_1e5])), 
batch_size=BATCH_SIZE, shuffle=True)


# %%
Loss_train = []
Loss_val  = []
for epoch in range(1, EPOCHS + 1):
    start_time = time.perf_counter()
    Loss_train.append(train(model, DEVICE, train_loader_1e5, optimizer, epoch))
    Loss_val.append(test(model, DEVICE, val_loader))
    # print(time.perf_counter() - start_time)

# %%
