# %%
import torch
import numpy as np

# %%
for i in range(20):
    d = torch.load("../result/regression/SimpleCNNRegression/epoch"+str(i)+".pth")
    y = d["y"]
    pred = d["pred"]
    idx = [idx for idx,each in enumerate(y) if each!=0]
    yp = [y[i] for i in idx]
    predp = [pred[i] for i in idx]
    print(np.mean([(yp[i]-predp[i])**2 for i in range(len(yp))]))


# %%
