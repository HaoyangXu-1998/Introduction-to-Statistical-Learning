import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from functools import wraps


class MeasureDataset(Dataset):
    def __init__(self, X, Y, annotate=""):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        self.annotate = annotate

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return "MeasureDataset: length=%d, annotate=%s" % (len(self),
                                                           self.annotate)


class load_data:
    Type_dir = "TypeIndex_batch"
    X_dir = "X_batch"
    Y_dir = "Y_batch"

    def __init__(self, directory, filename):
        self.dir = directory
        self.filepath = filename

        print("begin to load X")
        self.X = np.load(self.X_path)
        print("begin to load Y")
        self.Y = np.load(self.Y_path)
        print("begin to load type")
        self.Type = np.load(self.Type_path)

    @property
    def X_path(self):
        return os.path.join(self.dir, self.X_dir, self.filepath)

    @property
    def Y_path(self):
        return os.path.join(self.dir, self.Y_dir, self.filepath)

    @property
    def Type_path(self):
        return os.path.join(self.dir, self.Type_dir, self.filepath)

    def transformX(self):
        # HWC2CHW
        print("begin to transform X")
        self.X = self.X.transpose(0,3,1,2)
        return self

    def transformY(self):
        print("begin to transform Y")
        self.Y = np.array([1 if each > 0 else 0 for each in self.Y])
        return self

    def filter_type(self, cond=lambda x: x == 1):
        print("begin to filter type")
        mask = [i for i, flag in enumerate(self.Type) if cond(flag)]
        self.Type = self.Type[mask]
        self.X = self.X[mask]
        self.Y = self.Y[mask]
        return self

    def filter_value(self, cond=lambda x: x > 0):
        print("begin to filter value")
        mask = [i for i, flag in enumerate(self.Y) if cond(flag)]
        self.Type = self.Type[mask]
        self.X = self.X[mask]
        self.Y = self.Y[mask]
        return self

    def filter_na(self):
        print("begin to filter na")
        maskx = [i for i,flag in enumerate(self.X) if any(flag.ravel()!=-9999)]
        masky = [i for i, flag in enumerate(self.Y) if flag != -9999]
        mask = np.unique(maskx+masky)
        self.Type = self.Type[mask]
        self.X = self.X[mask]
        self.Y = self.Y[mask]
        return self

    def split(self, test_size=0.2, seed=0):
        print("begin to split dataset")
        length = len(self.Y)
        mask_train, mask_valid, _, _ = train_test_split(list(range(length)),
                                                        self.Y,
                                                        test_size=test_size,
                                                        random_state=seed)
        return MeasureDataset(self.X[mask_train], self.Y[mask_train],
                              "train"), MeasureDataset(self.X[mask_valid],
                                                       self.Y[mask_valid],
                                                       "valid")
    
    def data(self):
        return MeasureDataset(self.X,self.Y)
