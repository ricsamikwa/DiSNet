
import sys
import numpy as np

SERVER_ADDR= '192.168.1.38'
SERVER_PORT = 43000

# Dataset 

N = 50000 # data length

# configuration settings - model
model_cfg = {
	# 'VGG' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	# ('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	# ('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	# ('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	# ('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	# ('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	# ('D', 128, 10, 1, 10, 128*10)]

	'VGG' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]

}

model_name = 'VGG'
model_size = 1.28
# split_layer = [2, 3, 2] 
split_layer = [6, 6] 
model_len = 7 #
DEVICE_PACE_RATE = 10
#also
rate = 10

# max parallelisable partitions in integers [1-4:5, 5-9:4, 10-14:3, 14-18:2]
max_par_partitions = [6,5,4,3] # based on the neural network DAG [6,5,4,3]
# max_par_partitions = [5,4,3,2]
pos_max_par_partitions = [4,9,12,18] # based on the neural network[4,8,12,18] [4,9,14,18]
layer_range = np.array([[0,4],[4,9],[9,14],[14,18]]) #very unreadable


R = 100 
LR = 0.01 # learning rate
B = 100 # minibatch size

# Network info
# K = 5
# HOST2IP = {'pi':'192.168.1.33' , 'nano2':'192.168.1.41', 'nano4':'192.168.1.40' , 'nano6':'192.168.1.42', 'nano8':'192.168.1.43'}
# CLIENTS_CONFIG= {'192.168.1.33':0, '192.168.1.41':1, '192.168.1.40':2, '192.168.1.42':3, '192.168.1.43':4}
# CLIENTS_LIST= ['192.168.1.33', '192.168.1.41', '192.168.1.40', '192.168.1.42', '192.168.1.43'] 
K = 2 

iteration = {'192.168.1.33':5, '192.168.1.41':5, '192.168.1.40':10, '192.168.1.42':5, '192.168.1.43':5}  

random = True
random_seed = 0