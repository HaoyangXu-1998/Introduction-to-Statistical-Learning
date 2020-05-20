import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from data import load_data, MeasureDataset
from model import SimpleCNN
from tqdm import tqdm
from sklearn.metrics import classification_report
from loss import FocalLoss

if __name__ == "__main__":
    data = load_data("data", "merge.npy")
    train_dataset, valid_dataset = data.filter_na().filter_type().\
                                   transformX().transformY().split()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=100,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=False,
                                  drop_last=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=100,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=False,
                                  drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = FocalLoss(classes=2,alpha=torch.FloatTensor([0.1,1]).to(device)).to(device)
    optimizer = opt.Adam(model.parameters(), lr=0.001)

    EPOCHNUM = 10
    for epoch in range(EPOCHNUM):

        with tqdm(total=len(train_dataloader)) as pbar:
            for idx, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                output = model(x.to(device))
                loss = criterion(output, y.to(device))
                pbar.set_description(
                    "training - Epoch %d Iter %d - loss %.4f" %
                    (epoch, idx, loss.item()))
                pbar.update(1)
                loss.backward()
                optimizer.step()

        pred_valid = []
        y_valid = []
        for idx, (x, y) in enumerate(valid_dataloader):
            output = model(x.to(device))
            pred = torch.argmax(output, 1)
            pred_valid += pred.detach().cpu().numpy().tolist()
            y_valid += y.detach().cpu().numpy().tolist()
        report = classification_report(y_valid, pred_valid)
        print(report)
