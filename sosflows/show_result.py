from prettytable import PrettyTable 
import pickle as pk
import torch

result_table = PrettyTable()
result_table.field_name  = ['Dataset',  'test-loss']
result_table.float_format = 0.3


# print table

# dataset = ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300']
dataset = ['POWER', 'GAS']

# load trained model
MODEL_DIR = "trained_models/"
output_file = MODEL_DIR + "result_table"

for name in dataset:
	path = MODEL_DIR + name
	# with open (path,'rb') as pickle_file:
	
	data_dic = torch.load(path)
	loss  = data_dic['test_loss']
	
	print(("{}_param:{}").format(name,data_dic['args']))
	print(("test_loss_{}:{}").format(name,loss))
	print("--------------------------------------")
	# result_table.add_row([name]+loss)
	
	# path = MODEL_DIR + name +"_m"
	# data_dic =torch.load(path)
	# loss =data_dic['test_loss']
	# print (loss)
	# result_table.add_row([name]+loss)

# print
# print (result_table)

# save table
# with open(output_file, mode = 'w') as f:
# 	 f.write(str(oc_table))

