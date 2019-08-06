import numpy as np
def modify_inf(result):
	# change nan to zero
	result[result!=result]=0

	# change inf to 10**20
	rsize = len(result)
	
	for i in range(0,rsize):
		
		if np.isinf(result[i]):
			if result[i]>0:
				result[i] = +10**20
			else:
				result[i] = -10**20
	return result