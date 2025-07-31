import pickle
import numpy as np
import pandas as pd
import os

address = os.getcwd().replace("\\", "/")
tail = "pkl"
path = address + "/AT_TCN_seed" + "/dataset/2DGesture/2DGesture_train"
with open(path + ".pkl", 'rb') as f:
    data = pickle.load(f)
df = pd.DataFrame(data)
df = df.iloc[:, :-1]
np.save(path + ".npy", df)

path = address + "/AT_TCN_seed" + "/dataset/2DGesture/2DGesture_test"
with open(path + ".pkl", 'rb') as f:
    data = pickle.load(f)
df = pd.DataFrame(data)
test = df.iloc[:, :-1] # tất cả các cột trừ cột cuối
labeled = df.iloc[:, -1].values   # chỉ cột cuối
np.save(path + ".npy", test)
np.save(path + "_label.npy", labeled)
print("done")

