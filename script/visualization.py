# %%
import sys
sys.path.append("..")
from data import load_data
import matplotlib.pyplot as plt
import numpy as np

# %%
data = load_data("../data", "merge.npy")
print("done")

# %%
data = data.filter_na().filter_type().filter_value()

# %%
x = data.X
x.shape

# %%
xnew = x[:,8,8,:]
xnew.shape

# %%
xlist = [each for each in xnew.ravel() if each>=0]
len(xlist)

# %%
histnum = 100
bins=np.arange(min(xlist),max(xlist),(max(xlist)-min(xlist))/histnum)
plt.hist(xlist,bins)
plt.xlabel('X')
plt.ylabel('count')
plt.show()


# %%
y = data.Y
y.shape


# %%
histnum = 100
bins=np.arange(min(y),max(y),(max(y)-min(y))/histnum)
plt.hist(y,bins)
plt.xlabel('Y')
plt.ylabel('count')
plt.show()

# %%
histnum = 100
bins=np.arange(min(y),max(y),(max(y)-min(y))/histnum)
plt.hist(y,bins)
plt.xlabel('Y')
plt.ylabel('count')
plt.show()

# %%
threshold = 0.5
yr = [each for each in y if each>threshold]
print(y.shape,len(yr))
histnum = 100
bins=np.arange(min(yr),max(yr),(max(yr)-min(yr))/histnum)
plt.hist(yr,bins)
plt.xlabel('Y without defined zero value')
plt.ylabel('count')
plt.show()

# %%
