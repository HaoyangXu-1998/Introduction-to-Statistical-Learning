# %%
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# %%
Y = np.load('../data/Y.npy')
X = np.load('../data/X.npy')
Land = np.load('../data/Land.npy')
mask_missing = Y>=0
Y = Y[mask_missing]
X = X[mask_missing]
Land = Land[mask_missing]
# %%
Y[Y>0] = 1
# %%
# Ocean specific
mask_ocean = Land==1
X_ocean = X[mask_ocean]
Y_ocean = Y[mask_ocean]

# %%
tmp = np.unique(np.where(X_ocean < 0)[0])
tmp = np.setdiff1d(np.arange(len(X_ocean)), tmp)
X_ocean = X_ocean[tmp]
Y_ocean = Y_ocean[tmp]
# %%
X_ocean = X_ocean[:, 7, 7, :]
# %%
np.random.seed(999)
idx = np.random.choice(len(Y_ocean), len(Y_ocean), False)

# %%
train_len = 423414
val_len = 12000
idx_train = idx[:train_len].reshape((9, -1))
 # %%
# ##############################
# SVM (toooooooooo slow)
# ##############################
from sklearn import svm
from sklearn.preprocessing import scale
# %%
weights = np.arange(start=1.0, stop=3.1, step=0.1)
SVM_forest = []
X_val_scaled = scale(X_ocean[idx[train_len:(train_len + val_len)], :], axis=0)
Y_val = Y_ocean[idx[train_len:(train_len + val_len)]]
# %%
for i in range(idx_train.shape[0]):
    scores = []
    X_train_scaled = scale(X_ocean[idx_train[i], :], axis=0)
    Y_train = Y_ocean[idx_train[i]]
    for weight in weights:
        clf = svm.SVC(cache_size=30000, class_weight={0.0: 1, 1.0: weight}).fit(X_train_scaled, Y_train)
        score = clf.score(X_val_scaled, Y_val)
        print("i = ", i, "Weight = ", weight, "Score = ", score)
        scores.append(score)
    SVM_forest.append(svm.SVC(cache_size=30000, 
                              class_weight={0: 1, 1: weights[np.argmax(scores)]}).fit(X_train_scaled, Y_train))


# %%
del X, Y, Land, mask_missing, mask_ocean


# %%
X_test_scaled = scale(X_ocean[idx[(423424 + 141269): ] ])
Y_test = Y_ocean[idx[(423424 + 141269): ] ]

# %%
# ##############################
# SVM Forest
# ##############################
pred = []
for m in SVM_forest:
    start_time = time.process_time()
    pred.append(m.predict(X_test_scaled))
    print(time.process_time() - start_time)
pred = np.array(pred)
# %%
for m in SVM_forest:
    print(m.score(X_test_scaled, Y_test))
# %%
pred = np.mean(pred, axis=0)
# %%
pred[pred > 0.5] = 1
pred[pred < 0.5] = 0
# %%
result_svm = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_svm = (result_svm[0, 0]) / (result_svm[0, 0] + result_svm[1, 0])
FAR_svm = 1 - (result_svm[0, 0])/(result_svm[0, 0] + result_svm[0, 1])



# %%
# ##############################
# EDA
# ##############################
plt.hist(Y[Y >= 0.1], bins=np.linspace(0.1, 100, 2000))
plt.xscale('log')
plt.show()
# %%
