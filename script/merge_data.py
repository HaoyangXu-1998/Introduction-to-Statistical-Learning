import os
import re
import numpy as np
from tqdm import tqdm

data_dir = "data"
batch_data_dir = ["TypeIndex_batch", "X_batch", "Y_batch"]

for batch_dir in batch_data_dir:
    path = os.path.join(data_dir, batch_dir)
    filepaths = sorted(os.listdir(path),key=lambda x: int(re.findall('\d+', x)[0]))
    cache = []
    for filepath in tqdm(filepaths, ncols=40):
        tmp = np.load(os.path.join(path, filepath))
        cache.append(tmp)
    output = np.concatenate(cache)
    np.save(os.path.join(path, "merge.npy"), output)
    print(output.shape)
