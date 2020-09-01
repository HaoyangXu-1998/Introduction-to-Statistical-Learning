# %%
import torch
import torch.nn as nn
import torch.optim as opt
import joblib
from torch.utils.data import DataLoader
import numpy as np
from data import load_data, MeasureDataset
from model import *
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_squared_error
from loss import FocalLoss, ExpandMSELoss, CombinedLoss
from utils import get_parameter_number, save_result

# %%
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("begin to load data")
with open("data/train.pt","rb") as f:
    train_dataset = joblib.load(f)
with open("data/valid.pt","rb") as f:
    valid_dataset = joblib.load(f)
# %%
model = SimpleCNNTwoTask().to(device)
print("model %s param number: %s" %
    (repr(model), get_parameter_number(model)))
strategy = {
    "clsf_begin_epoch":0,"clsf_end_epoch":10,
    "reg_begin_epoch":10,"reg_end_epoch":20,
    "mix_begin_epoch":20,"mix_end_epoch":35,
    }
reg_loss = nn.MSELoss()
clsf_loss = nn.CrossEntropyLoss()
criterion = CombinedLoss(reg_loss=reg_loss, clsf_loss=clsf_loss, strategy=strategy).to(device)
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
            loss,loss_type = criterion(clsf, reg, y.to(device), epoch, True)
            loss_train.append(loss.item())
            pbar.set_description(
                "training - Epoch %d Iter %d - loss %.4f - type %s " %
                (epoch, idx, loss.item(),loss_type))
            pbar.update(1)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(),
            #                             max_norm=20,
            #                             norm_type=2)
            optimizer.step()

    model.eval()
    pred_reg_valid = []
    pred_clsf_valid = []
    y_reg_valid = []
    y_clsf_valid = []
    loss_valid = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(valid_dataloader):
            clsf, reg = model(x.to(device))
            loss,loss_type = criterion(clsf, reg, y.to(device), epoch, False)
            loss_valid.append(loss.item())
            pred_clsf_valid += torch.argmax(clsf,1).detach().cpu().numpy().tolist()
            pred_reg_valid += reg.detach().cpu().numpy().tolist()
            y_clsf_valid += y[:, 0].detach().cpu().numpy().tolist()
            y_reg_valid += y[:, 1].detach().cpu().numpy().tolist()

    report = classification_report(y_clsf_valid, pred_clsf_valid)
    print(report)
    mse = mean_squared_error(y_reg_valid, pred_reg_valid)
    base = mean_squared_error(y_reg_valid, [torch.Tensor(y_reg_valid).mean().item()] * len(y_reg_valid))
    rate = mse / base
    print("loss type:%s mse:%.4f,rate of baseline:%.4f" % (loss_type, mse, rate))

    # losses = {"train": np.mean(loss_train), "valid": np.mean(loss_valid)}
    # save_result(TASK, model, epoch, y_valid, pred_valid, losses)