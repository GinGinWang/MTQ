import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from result_helpers.utils import *
import pickle

model_name ="LSA_SOS"
method_name = "mul"
class_name = "2"

data_path = 'checkpoints/mnist/8LSA_1_loss_history.npz'
with np.load(data_path, allow_pickle = True) as data:

	loss_history= data['loss_history']
	train_loss = loss_history.item().get('train_loss')
	train_rec = loss_history.item().get('train_rec')
	train_nllk = loss_history.item().get('train_nllk')

	validation_loss =  loss_history.item().get('validation_loss')
	# validation_rec = loss_history.item().get('validation_rec')
	# validation_nllk =  loss_history.item().get('validation_nllk')

	best_validation_epoch = data['best_validation_epoch'] 
	best_validation_rec_epoch= data['best_validation_rec_epoch']


print (best_validation_epoch)
print (best_validation_rec_epoch)

x = range(0,len(train_loss),1)
fig = plt.figure(0)
# plt.subplot(311)
l1 = plt.plot(x, train_loss, 'b',label = 'train_loss')
plt.plot(x, validation_loss, 'r',label = 'validation_loss')
plt.ylim(0,10)
# plt.subplot(312)
# l2 = plt.plot(x, train_rec, 'b',label = 'train_rec')
# plt.plot(x, validation_rec, 'r',label = 'validation_rec')

# plt.subplot(313)
# l3 = plt.plot(x, train_nllk, 'b',label = 'train_nllk')
# plt.plot(x, validation_nllk, 'r',label = 'validation_nllk')

# plt.figlegend.legend((l1, l2, l3),('loss','rec','nllk'), 'upper right')
plt.savefig(f'distgraph/lsa_loss_history.png')

