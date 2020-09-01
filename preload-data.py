import joblib
from data import load_data, MeasureDataset

data = load_data("data", "merge.npy")
train_dataset, valid_dataset, test_dataset = data.filter_na().filter_type().transformX().transformY_2D().split()
with open("data/train.pt","wb") as f:
    joblib.dump(train_dataset,f)
with open("data/valid.pt","wb") as f:
    joblib.dump(valid_dataset,f)
with open("data/test.pt","wb") as f:
    joblib.dump(test_dataset,f)