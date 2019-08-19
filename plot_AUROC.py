import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from result_helpers.utils import *
import pickle

model_name ="LSA_SOS"
train_strategy = "mul"
class_name = "2"

xmin = 0
xmax = 3000
ymin = 0.9
ymax = 1.0


file_name = f'c{class_name}_{model_name}_{train_strategy}_auroc'

pickle_in = open(file_name,"rb")
auroc = pickle.load(pickle_in)

print(auroc.keys())
auroc_len = len(auroc['ns'])

x = range(0, auroc_len*50, 50)

fig = plt.figure(0)
plt.axis([xmin, xmax, ymin, ymax])

plt.plot(x, auroc['ns'], 'bo-',label = 'ns')
plt.plot(x, auroc['rec'], 'g--',label = 'rec')
plt.plot(x, auroc['nllk'], 'r>-',label = 'nllk')
plt.plot(x, auroc['q1'], 'co-',label = 'q1')
plt.plot(x, auroc['q2'], 'm--',label = 'q2')
plt.plot(x, auroc['qinf'], 'y<-',label = 'qinf')

plt.legend(loc='lower right')



# Method 1

plt.savefig(f'distgraph/{file_name}.png')
