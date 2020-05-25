# %%
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import numpy as np
from data import load_data, MeasureDataset
from model import SimpleCNNClassification, SimpleCNNRegression
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_squared_error
from loss import FocalLoss, ExpandMSELoss
from utils import get_parameter_number, save_result
import matplotlib.pyplot as plt
# %%
TASK = "regression"
filter_0 = False
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("begin to load data")
    data = load_data("data", "merge.npy")
    if TASK == "classification":
        train_dataset, valid_dataset = data.filter_na().filter_type().\
                                   transformX().transformY().split()
        model = SimpleCNNClassification().to(device)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
        print("model %s param number: %s" %
              (repr(model), get_parameter_number(model)))
        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = FocalLoss(classes=2,         
                              alpha=torch.FloatTensor([1,10]).to(device),
                              size_average=True).to(device)
        optimizer = opt.Adam(model.parameters(), lr=2e-5)
    elif TASK == "regression":
        # Don't filter value in data, but create 
        # another Dataset
        train_dataset, valid_dataset = data.filter_na().filter_type().\
                                   transformX().split()
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
        criterion = nn.MSELoss().to(device)
        #criterion = ExpandMSELoss().to(device)
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
    # EPOCHNUM = 25
    # for epoch in range(EPOCHNUM):
    #     model.train()
    #     loss_train = []
    #     with tqdm(total=len(train_dataloader)) as pbar:
    #         for idx, (x, y) in enumerate(train_dataloader):
    #             optimizer.zero_grad()
    #             output, _, _ = model(x.to(device))
    #             loss = criterion(output, 
    #             y.long().to(device) if TASK == "classification" else y.to(device))
    #             loss_train.append(loss.item())
    #             pbar.set_description(
    #                 "training - Epoch %d Iter %d - loss %.4f" %
    #                 (epoch, idx, loss.item()))
    #             pbar.update(1)
    #             loss.backward()
    #             nn.utils.clip_grad_norm_(model.parameters(),
    #                                      max_norm=20,
    #                                      norm_type=2)
    #             optimizer.step()

    #     model.eval()
    #     pred_valid = []
    #     y_valid = []
    #     catt_valid = []
    #     satt_valid = []
    #     loss_valid = []
    #     for idx, (x, y) in enumerate(valid_dataloader):
    #         output, catt, satt = model(x.to(device))
    #         loss = criterion(output, y.long().to(device) if TASK == "classification" else y.to(device))
    #         loss_valid.append(loss.item())
    #         catt_valid.append(catt)
    #         satt_valid.append(satt)
    #         if TASK == "classification":
    #             pred = torch.argmax(output, 1)
    #             pred_valid += pred.detach().cpu().numpy().tolist()
    #             y_valid += y.detach().cpu().numpy().tolist()
    #         elif TASK == "regression":
    #             pred_valid += output.detach().cpu().numpy().tolist()
    #             y_valid += y.detach().cpu().numpy().tolist()

    #     if TASK == "classification":
    #         report = classification_report(y_valid, pred_valid)
    #         print(report)
    #     elif TASK == "regression":
    #         mse = mean_squared_error(y_valid, pred_valid)
    #         base = mean_squared_error(y_valid, [torch.Tensor(y_valid).mean().item()] * len(y_valid))
    #         rate = mse / base
    #         print("mse:%.4f,rate of baseline:%.4f" % (mse, rate))

    #     losses = {"train": np.mean(loss_train), "valid": np.mean(loss_valid)}
    #     attentions = {
    #         "channel": torch.cat(catt_valid,0).detach().cpu().numpy(),
    #         "spatial": torch.cat(satt_valid,0).detach().cpu().numpy()
    #     }
    #     save_result(TASK, model, epoch, y_valid, pred_valid, losses, attentions)


# %%
    # No attention
    EPOCHNUM = 25
    for epoch in range(EPOCHNUM):
        model.train()
        loss_train = []
        with tqdm(total=len(train_dataloader)) as pbar:
            for idx, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                output = model(x.to(device))
                loss = criterion(output, 
                y.long().to(device) if TASK == "classification" else y.to(device))
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
            output = model(x.to(device))
            loss = criterion(output, y.long().to(device) if TASK == "classification" else y.to(device))
            loss_valid.append(loss.item())
            if TASK == "classification":
                pred = torch.argmax(output, 1)
                pred_valid += pred.detach().cpu().numpy().tolist()
                y_valid += y.detach().cpu().numpy().tolist()
            elif TASK == "regression":
                pred_valid += output.detach().cpu().numpy().tolist()
                y_valid += y.detach().cpu().numpy().tolist()

        if TASK == "classification":
            report = classification_report(y_valid, pred_valid)
            print(report)
        elif TASK == "regression":
            mse = mean_squared_error(y_valid, pred_valid)
            base = mean_squared_error(y_valid, [torch.Tensor(y_valid).mean().item()] * len(y_valid))
            rate = mse / base
            print("mse:%.4f,rate of baseline:%.4f" % (mse, rate))

        losses = {"train": np.mean(loss_train), "valid": np.mean(loss_valid)}
        save_result(TASK, model, epoch, y_valid, pred_valid, losses)
