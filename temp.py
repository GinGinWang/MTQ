from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as io



path='data/UCI'
# # # KDDCup 10% Data
# url_data = f"{path}/kddcup.data_10_percent.gz"
# # info data (column names, col types)
# url_info = f"{path}/kddcup.names"

# # Import info data
# df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
# colnames = df_info.colname.values
# print(colnames)
# coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
# colnames = np.append(colnames, ["status"])
# coltypes = np.append(coltypes, ["str"])

# print(colnames)
# print(colnames.shape)
# print(coltypes)
# print(coltypes.shape)
# # Import data
# df = pd.read_csv(url_data, names=colnames, index_col=False,
#                  dtype=dict(zip(colnames, coltypes)))
# # Dumminize

# X = pd.get_dummies(df.iloc[:,:-1]).values

# X = pd.get_dummies(df).values

# print(df.head())
# print(X[0,:])
# print(X[1,:].shape)
# print(X[2,:].shape)


#

print(max(X[:,7]))
print(min(X[:,7]))
print(max(X[:,11]))
print(min(X[:,11]))
print(max(X[:,21]))
print(min(X[:,21]))
print(max(X[:,22]))
print(min(X[:,22]))
print(X.shape)
print(X[0,:])
print(y[0])
