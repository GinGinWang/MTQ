import numpy as np 
import os 
from torchvision import datasets

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# dataset = 'fashion-mnist'

dataset ='cifar10'

# ft_dir = '/home/jj27wang/NewSoSLSA/baseline_ADT/gtfeatures'
ft_dir = '/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/baseline_ADT/gtfeatures'
data_dir = f'/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/data/{dataset}-gth'

for single_class in range(1):
	print(single_class)
	for epoch in [0]:
		train_data_name = f'{dataset}_{epoch:02d}_gth_{single_class}_train'
		# for ft_class in range(72):
		# 	feature_name =f'train_e{epoch:02d}_{dataset}_m{single_class}_d{single_class}_t{ft_class}.npz'
		# 	feature = np.load(os.path.join(ft_dir, feature_name))['arr_0']
		# 	featurelist.append(feature.reshape(int(feature.shape[0]*feature.shape[1]/256),256))
			
		feature_name =f'train_e{epoch:02d}_{dataset}_m{single_class}.npz'
		featurelist   = np.load(os.path.join(ft_dir, feature_name))['arr_0']
		# 
		featurelist = np.array(featurelist)
		print(featurelist.shape)

		featurelist = featurelist.transpose(1,0,2)

		print(featurelist.shape)

		np.savez(os.path.join(data_dir,train_data_name),featurelist)


for single_class in range(1):
	print(single_class)
	for epoch in [0]:
		new_data ={}
		
		train_data_name = f'{dataset}_{epoch:02d}_gth_{single_class}_train.npz'
		train_data_name2 = f'{dataset}_{epoch:02d}_gth_{single_class}_train_xy.npy'

		data = np.load(os.path.join(data_dir,train_data_name))['arr_0']
		
		lenth = len(data)
		for i in range(lenth):
			new_data[i] = [data[i,:,:],single_class]
		
		np.save(os.path.join( data_dir,train_data_name2),new_data)




# merge features
for single_class in range(1):

	print(data_class)
	for epoch in [0]:
		featurelist =[]
		test_data_name = f'{dataset}_{epoch:02d}_m{single_class}_test'
		for ft_class in range(72):
			feature_name = f'test_e{epoch:02d}_{dataset}_m{single_class}_d{data_class}_t{ft_class}.npz'

			feature = np.load(os.path.join(ft_dir, feature_name))['arr_0']
			# print(feature.shape)

			featurelist.append(feature.reshape(int(feature.shape[0]*feature.shape[1]/256),256))
			print(np.array(featurelist).shape)

		featurelist = np.array(featurelist)
		featurelist = featurelist.transpose(1,0,2)
		print(featurelist.shape)

		np.savez(os.path.join(data_dir,test_data_name),featurelist)



# for single_class in range(1):
# 	for epoch in [1,2,3]:
# 		new_test_data_name = f'{dataset}_m{single_class}_e{epoch:02d}_gth_test.npy'
# 		new_data ={}
# 		for data_class in range(10):
# 			test_data_name = f'{dataset}_{epoch:02d}_m{single_class}_d{data_class}_test.npz'
# 			print(test_data_name)
# 			data = np.load(os.path.join(data_dir,test_data_name))['arr_0']
# 			print(data.shape)
# 			lenth = len(data)
# 			for i in range(lenth):
# 				new_data[data_class*lenth+i] = [data[i,:,:],data_class]
# 				print (new_data[data_class*lenth+i][0].shape)
# 				print(new_data[data_class*lenth+i][1])

# 		np.save(os.path.join(data_dir,new_test_data_name),new_data)

