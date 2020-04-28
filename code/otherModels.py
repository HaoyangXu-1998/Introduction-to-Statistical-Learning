# %%
import numpy as np
import os
import time
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
X = np.load('../data/X.npy')
Y = np.load('../data/Y.npy')
Land = np.load('../data/Land.npy')
# %%
mask_missing = Y>=0
Y = Y[mask_missing]
X = X[mask_missing]
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
Y[Y>0] = 1
# %%
# Ocean specific
mask_ocean = Land[mask_missing]==1
X_ocean = X[mask_ocean]
Y_ocean = Y[mask_ocean]
# %%
tmp = np.unique(np.where(X_ocean < 0)[0])
tmp = np.setdiff1d(np.arange(len(X_ocean)), tmp)
X_ocean = X_ocean[tmp]
Y_ocean = Y_ocean[tmp]
# %%
np.random.seed(999)
idx = np.random.choice(len(Y_ocean), len(Y_ocean), False)

# %%
train_len = 423424
val_len = 141269
X_train = X_ocean[idx[:train_len], 7, 7, :]
X_val   = X_ocean[idx[train_len:(train_len+val_len)], 7, 7, :]
X_test  = X_ocean[idx[(train_len+val_len):], 7, 7, :]
Y_train = Y_ocean[idx[:train_len]]
Y_val   = Y_ocean[idx[train_len:(train_len+val_len)]]
Y_test  = Y_ocean[idx[(train_len+val_len):]]
# %%
del X, Y, Land, mask_missing, mask_ocean
# %%
#############################
# Logistic Regression
#############################
from sklearn.linear_model import LogisticRegression
# %%
for i in [0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04, 1.06, 1.08, 1.1]:
    clf = LogisticRegression(class_weight = {0: 1, 1: i}).fit(X_train, Y_train)
    print('i = ', i, "Score = ", clf.score(X_val, Y_val))


# %%
clf = LogisticRegression(class_weight = {0: 1, 1: 1.02}).fit(X_train, Y_train)
# %%
pred = clf.predict(X_test)


# %%
result_logit = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_logit = (result_logit[0, 0]) / (result_logit[0, 0] + result_logit[1, 0])
FAR_logit = 1 - (result_logit[0, 0])/(result_logit[0, 0] + result_logit[0, 1])

# %%
###############################
# Ridge Classification
###############################
from sklearn.linear_model import RidgeClassifier
# %%
for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
    for i in [0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]:
        clf = RidgeClassifier(alpha=alpha, class_weight={0:1, 1:i}).fit(X_train, Y_train)
        print('alpha = ', alpha, "i = ", i, "Score = ", clf.score(X_val, Y_val))
    

# %%
clf = RidgeClassifier(class_weight={0:1, 1:2}).fit(X_train, Y_train)
pred = clf.predict(X_test)
# %%
result_ridge = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_ridge = (result_ridge[0, 0]) / (result_ridge[0, 0] + result_ridge[1, 0])
FAR_ridge = 1 - (result_ridge[0, 0])/(result_ridge[0, 0] + result_ridge[0, 1])

# %%
###############################
# SVM (toooooooooo slow)
###############################
# from sklearn import svm
# from sklearn.preprocessing import scale
# # %%
# X_train_scaled = scale(X_train)
# X_val_scaled   = scale(X_val)
# X_test_scaled  = scale(X_test)
# # %%
# for i in [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
#     clf = svm.SVC(cache_size=20000, class_weight={0: 1, 1: i}).fit(X_train_scaled, Y_train)
#     print('i = ', i, "Score = ", clf.score(X_val_scaled, Y_val))

# %%
##############################
# KNN
##############################
from sklearn.neighbors import KNeighborsClassifier
# %%
for k in range(3, 20):
    neigh = KNeighborsClassifier(k, n_jobs=-1).fit(X_train, Y_train)
    print("k = ", k, "Score = ", neigh.score(X_val, Y_val))

# %%
neigh = KNeighborsClassifier(17, n_jobs=-1)
neigh.fit(X_train, Y_train)
# %%
pred = neigh.predict(X_test)
# %%
result_knn = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_knn = (result_knn[0, 0]) / (result_knn[0, 0] + result_knn[1, 0])
FAR_knn = 1 - (result_knn[0, 0])/(result_knn[0, 0] + result_knn[0, 1])

# %%
#############################
# Gaussian Naive Bayes (bad)
#############################
from sklearn.naive_bayes import GaussianNB
# %%
clf = GaussianNB(priors=[0.89, 0.11])
clf.fit(X_train, Y_train)

# %%
pred = clf.predict(X_test)
# %%
result_GNB = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_GNB = (result_GNB[0, 0]) / (result_GNB[0, 0] + result_GNB[1, 0])
FAR_GNB = 1 - (result_GNB[0, 0])/(result_GNB[0, 0] + result_GNB[0, 1])

# %%
#############################
# Complement Naive Bayes 
#############################
from sklearn.naive_bayes import ComplementNB

# %%
tmp = np.unique(np.where(X_train > 0)[0])
X_train_NB = X_train[tmp, :]
Y_train_NB = Y_train[tmp]
# %%
tmp = np.unique(np.where(X_test > 0)[0])
X_test_NB  = X_test[tmp, :]
Y_test_NB  = Y_test[tmp]
# %%
clf = ComplementNB()
clf.fit(X_train_NB, Y_train_NB)

# %%
pred = clf.predict(X_test)
# %%
result_NB = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_NB = (result_NB[0, 0]) / (result_NB[0, 0] + result_NB[1, 0])
FAR_NB = 1 - (result_NB[0, 0])/(result_NB[0, 0] + result_NB[0, 1])

# %%
from sklearn.ensemble import RandomForestClassifier
# %%
clf = RandomForestClassifier(max_depth=50)
start_time = time.process_time()
clf.fit(X_train, Y_train)
print(time.process_time() - start_time)
start_time = time.process_time()
print(clf.score(X_val, Y_val))
print(time.process_time() - start_time)
# %%
Scores = []
# 5 -> 50; 25 -> 35; 33 -> 37  
for depth in np.arange(start=33, stop=38, step=1):
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, Y_train)
    score = clf.score(X_val, Y_val)
    print("max_depth = ", depth, "score = ", score)
    Scores.append(score)

# %%
clf = RandomForestClassifier(max_depth=35)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
# %%
result_rf = np.array([[np.sum(pred * Y_test), np.sum(pred * (1 - Y_test))], [np.sum((1 - pred) * Y_test), np.sum((1 - pred) * (1 - Y_test))]])

# %%
POD_rf = (result_rf[0, 0]) / (result_rf[0, 0] + result_rf[1, 0])
FAR_rf = 1 - (result_rf[0, 0])/(result_rf[0, 0] + result_rf[0, 1])

# %%
