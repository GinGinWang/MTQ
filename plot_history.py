import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from result_helpers.utils import *
import pickle

model_name = "LSA"
train_strategy = "1"
class_name_list = ["1"]
# class_name_list = ['0','2','3','4','5','6','7','8','9']

dataset_name = "thyroid"
log_step = 100

for class_name in class_name_list:
	loss_file_name = f'{class_name}{model_name}_{train_strategy}_loss_history'
	data_path = f'checkpoints/{dataset_name}/{loss_file_name}.npz'
	if model_name in ["LSA_SOS", "LSA_MAF"]:
		data_path = f'checkpoints/{dataset_name}/b1h2048c64/{loss_file_name}.npz'

	with np.load(data_path, allow_pickle = True) as data:

		loss_history= data['loss_history']
		train_loss = loss_history.item().get('train_loss')
		train_rec = loss_history.item().get('train_rec')
		train_nllk = loss_history.item().get('train_nllk')

		validation_loss =  loss_history.item().get('validation_loss')
		validation_rec = loss_history.item().get('validation_rec')
		validation_nllk =  loss_history.item().get('validation_nllk')

		best_validation_epoch = data['best_validation_epoch'] 
		best_validation_rec_epoch= data['best_validation_rec_epoch']


	print (best_validation_epoch)
	print (best_validation_rec_epoch)

	x = range(0,len(train_loss),1)
	fig = plt.figure(0)

	ax1 =plt.subplot(311)
	ax1.plot(x, train_loss, 'b',label = 'train_loss')
	ax1.plot(x, validation_loss, 'r',label = 'validation_loss')
	ax1.legend(loc= 'upper left')
	ax1.set_ylabel('loss')

	# ax2 = plt.subplot(312)
	# ax2.plot(x, train_rec, 'b',label = 'train_rec')
	# ax2.plot(x, validation_rec, 'r',label = 'validation_rec')
	# ax2.legend(loc= 'upper left')

	# ax3 = plt.subplot(313)
	# ax3.plot(x, train_nllk, 'b',label = 'train_nllk')
	# ax3.plot(x, validation_nllk, 'r',label = 'validation_nllk')
	# ax3.legend(loc= 'upper left')

	# fig.suptitle(f"loss-history with {train_strategy} training strategy")

	file_name = f'{dataset_name}_c{class_name}_{model_name}_{train_strategy}_auroc'

	pickle_in = open(file_name,"rb")
	auroc = pickle.load(pickle_in)

	print(auroc.keys())
	auroc_len = len(auroc['ns'])
	x = range(0, auroc_len*log_step, log_step)
	ax4 = ax1.twinx()  # this is the important function
	ax4.plot(x, auroc['nllk'], 'go-',label = 'nllk-auroc')
	ax4.set_ylabel('AUROC')
	ax4.legend(loc= 'lower right')
	# ax4.set_ylim((0.9,1))

	plt.savefig(f'distgraph/{dataset_name}_{loss_file_name}.png')
	plt.close(0)

	# fig = plt.figure(1)
	# # plt.axis([xmin, xmax, ymin, ymax])

	# plt.plot(x, auroc['ns'], 'bo-',label = 'ns')
	# plt.plot(x, auroc['rec'], 'g--',label = 'rec')
	# plt.plot(x, auroc['nllk'], 'r>-',label = 'nllk')



	# if model_name in ['LSA_MAF','LSA_SOS']:
	# 	plt.plot(x, auroc['q1'], 'co-',label = 'q1')
	# 	plt.plot(x, auroc['q2'], 'm--',label = 'q2')
	# 	plt.plot(x, auroc['qinf'], 'y<-',label = 'qinf')

	# plt.legend(loc='lower right')


	# # plot_AUROC
	# plt.savefig(f'distgraph/{dataset_name}_{file_name}.png')
	# plt.close(1)
