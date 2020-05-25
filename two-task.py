# %%
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import numpy as np
from data import load_data, MeasureDataset
from model import *
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_squared_error
from loss import FocalLoss, ExpandMSELoss, CombinedLoss
from utils import get_parameter_number, save_result
import matplotlib.pyplot as plt

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("begin to load data")
data = load_data("data", "merge.npy")
train_dataset, valid_dataset = data.filter_na().filter_type().transformX().transformY_2D().split()
# %%
model = SimpleCNNTwoTask().to(device)
print("model %s param number: %s" %
    (repr(model), get_parameter_number(model)))
criterion = CombinedLoss(alpha=1).to(device)
optimizer = opt.Adam(model.parameters(), lr=1e-4)

# %%
train_dataloader = DataLoader(train_dataset,
                                batch_size=500,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=False)
valid_dataloader = DataLoader(valid_dataset,
                                batch_size=256,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)

print("training dataset size is %d, batch num is %d" %
        (len(train_dataset), len(train_dataloader)))
print("valid dataset size is %d, batch num is %d" %
        (len(valid_dataset), len(valid_dataloader)))

# %%
EPOCHNUM = 25
for epoch in range(EPOCHNUM):
    model.train()
    loss_train = []
    with tqdm(total=len(train_dataloader)) as pbar:
        for idx, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            clsf, reg = model(x.to(device))
            loss = criterion(clsf, reg, y.to(device))
            loss_train.append(loss.item())
            pbar.set_description(
                "training - Epoch %d Iter %d - loss %.4f" %
                (epoch, idx, loss.item()))
            pbar.update(1)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(),
            #                             max_norm=20,
            #                             norm_type=2)
            optimizer.step()

    model.eval()
    pred_valid = []
    y_valid = []
    loss_valid = []
    for idx, (x, y) in enumerate(valid_dataloader):
        clsf, reg = model(x.to(device))
        loss = criterion(clsf, reg, y.to(device))
        loss_valid.append(loss.item())
        pred_valid += reg.detach().cpu().numpy().tolist()
        y_valid += y[:, 1].detach().cpu().numpy().tolist()

    # report = classification_report(y_valid, pred_valid)
    # print(report)
    mse = mean_squared_error(y_valid, pred_valid)
    base = mean_squared_error(y_valid, [torch.Tensor(y_valid).mean().item()] * len(y_valid))
    rate = mse / base
    print("mse:%.4f,rate of baseline:%.4f" % (mse, rate))

    # losses = {"train": np.mean(loss_train), "valid": np.mean(loss_valid)}
    # save_result(TASK, model, epoch, y_valid, pred_valid, losses)

# %%
