# data_preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as io
# data = io.loadmat('./data/UCI/thyroid.mat')
# print(len(data['X']))
# print(data.keys())
# print(data['__header__'])
# print(data['__version__'])
# print(data['__globals__'])

# train_split ={}
# test_split = {}
# idxs = np.arange(len(data['y']))
# np.random.shuffle(idxs)

# train_split['X'] = data['X'][idxs[0:int(0.5*len(idxs))]]
# train_split['y'] = data['y'][idxs[0:int(0.5*len(idxs))]]

# print (train_split['X'][0])

# test_split['X'] = data['X'][idxs[int(0.5*len(idxs)):]]
# test_split['y'] = data['y'][idxs[int(0.5*len(idxs)):]]

# print (train_split['X'][0].shape)


url_base = "./data/UCI"

# # KDDCup 10% Data
url_data = f"{url_base}/kddcup.data_10_percent.gz"
# info data (column names, col types)
url_info = f"{url_base}/kddcup.names"

# Import info data
df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
colnames = df_info.colname.values
coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
colnames = np.append(colnames, ["status"])
coltypes = np.append(coltypes, ["str"])

# Import data
df = pd.read_csv(url_data, names=colnames, index_col=False,
                 dtype=dict(zip(colnames, coltypes)))


# Dumminize
X = pd.get_dummies(df.iloc[:,:-1]).values

# Create Traget Flag
# Anomaly data when status is normal, Otherwise, Not anomaly.
y = np.where(df.status == "normal.", 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=123)
X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]

print(len(X_train))

val_num = int(0.1*len(X_train))
print(val_num)

X_val = X_train[0:val_num,:]
y_val = y_train[0:val_num]
print(X.shape)
print (sum(y))
print(len(y))
x = X_test[0,:]

x = np.expand_dims(x, axis=0)
x = x.reshape((11,11))
x = np.expand_dims(x, axis=2)

print(x.shape)