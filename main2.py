# %%
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import numpy as np
from data import load_data, MeasureDataset
from model import SimpleCNNClassification, SimpleCNNRegression,InceptionTwoTask
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_squared_error
from loss import FocalLoss, ExpandMSELoss
from utils import get_parameter_number, save_result
import matplotlib.pyplot as plt

# %%
filter_0 = False
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("begin to load data")
    data = load_data("data", "merge.npy")
    # Don't filter value in data, but create 
    # another Dataset
    train_dataset, valid_dataset = data.filter_na().filter_type().\
                               transformX().transformY_2D().split()
    if filter_0:
        mask_train = [i for i, flag in enumerate(train_dataset.Y) if flag > 0]
        train_dataset = MeasureDataset(train_dataset.X[mask_train], train_dataset.Y[mask_train])
        mask_valid = [i for i, flag in enumerate(valid_dataset.Y) if flag > 0]
        valid_dataset = MeasureDataset(valid_dataset.X[mask_valid], valid_dataset.Y[mask_valid])

    model = SimpleCNNRegression().to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")

    print("model %s param number: %s" %
          (repr(model), get_parameter_number(model)))
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
    criterion = FocalLoss(classes=2,alpha=torch.FloatTensor([1,1]).to(device)).to(device)
    optimizer = opt.Adam(model.parameters(), lr=5e-4)
    EPOCHNUM = 10
    for epoch in range(EPOCHNUM):
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:
            for idx, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                score,output = model(x.to(device))
                loss = criterion(score,y[:,0].long().to(device)) 
                pbar.set_description(
                    "training - Epoch %d Iter %d - loss %.4f" %
                    (epoch, idx, loss.item()))
                pbar.update(1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=20,
                                         norm_type=2)
                optimizer.step()
        model.eval()
        pred_valid = []
        y_valid = []
        with tqdm(total=len(valid_dataloader)) as pbar:
            for idx, (x, y) in enumerate(valid_dataloader):
                score,output = model(x.to(device))
                pr = torch.argmax(score,1)
                pred_valid += pr.detach().cpu().numpy().tolist()
                y_valid += y[:,0].detach().cpu().numpy().tolist()
        print(classification_report(y_valid,pred_valid))



# %%
    # No attention
    criterion2 = ExpandMSELoss().to(device)
    optimizer = opt.Adam(model.parameters(), lr=1e-4,weight_decay=1e-2)
    EPOCHNUM = 20
    for epoch in range(EPOCHNUM):
        model.train()
        loss_train = []
        with tqdm(total=len(train_dataloader)) as pbar:
            for idx, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                score,output = model(x.to(device))
                loss = criterion2(output,y[:,1].to(device))
                loss_train.append(loss.item())
                pbar.set_description(
                    "training - Epoch %d Iter %d - loss %.4f" %
                    (epoch, idx, loss.item()))
                pbar.update(1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=20,
                                         norm_type=2)
                optimizer.step()

        model.eval()
        pred_valid = []
        y_valid = []
        loss_valid = []
        for idx, (x, y) in enumerate(valid_dataloader):
            score,output = model(x.to(device))
            loss = criterion2(output,y[:,1].to(device))
            loss_valid.append(loss.item())
            pred_valid += output.detach().cpu().numpy().tolist()
            y_valid += y[:,1].detach().cpu().numpy().tolist()

        mse = mean_squared_error(y_valid, pred_valid)
        base = mean_squared_error(y_valid, [torch.Tensor(y_valid).mean().item()] * len(y_valid))
        rate = mse / base
        print("mse:%.4f,rate of baseline:%.4f" % (mse, rate))

        losses = {"train": np.mean(loss_train), "valid": np.mean(loss_valid)}
        save_result("regression", model, epoch, y_valid, pred_valid, losses)
