import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import math
from field_vgg import FieldCalculator
from field_DiSNet import FieldCalculatorDiSNet
from torchvision import transforms
from models.model_vgg16 import VGG16
from inference import *
from utils import *
from network import *
import configurations



# load image
filename = ("input/dog.jpg")
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# setup parameters
# trans_rate = [40, 40, 40, 40] # Mbps
# num_sever = [3,4,5,6,7,8,9,10]
# num_sever = [3]

# load model
model = VGG16()
model_dict = model.state_dict()
model_dict.update(torch.load("opt/vgg16-modify.pth"))
model.load_state_dict(model_dict)
model.eval()


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda:0')
    model.to('cuda:0')


# inference DiSNet *** 
"""
max possible (gainful) parallelisable parts in integers [eg. layer 1 - 4: 5, layer 5 - 7: 4, ... , 1]
get the suitable path (more resources - longest path with less nodes) 
    - amortized cost of the path - path with highest average - uniform contribution for transmission and  comp rate e.g /5 
for each device in the path 
    get the neighbourhood resources info
at the current (max) parallelisable block
    get the subset (subset number = parallel blocks, ) of the neighbours with max possible resources
when at start node
    if the cost of transferring to the neighbours with max possible resources plus execution is better than the current neighbourhood
    move and start there
for the rest repeat
    at the current neighbours find a subset for the next (max) parallelisable block
    for the remaining neighbours in the path 
        find the neighbourhood with max possible resources
    if the cost of transferring to the neighbours with max possible resources plus execution is better than the current neighbourhood
    move and start there
for the last layer 
    check the remaining path
    execute it on the device with the highers resources 
"""


####### updates

# max parallelisable partitions in integers [1-4:5, 5-9:4, 10-14:3, 14-18:2]
max_par_partitions = configurations.max_par_partitions 

print("==================Initiating tests===================>")
# generate mesh network graph with 10 devices and 15 connections with random resources and transmission throughput

num_devices = 10
num_connections = 15
print('Generating random network graph of heterogenous resources (comp, network)')
G = generate_random_graph(num_devices, num_connections)
draw_graph(G)


#different nodes
while True:
    input_node = random.randint(0, 9)
    output_node = random.randint(0, 9)
    if input_node != output_node:
        break
print('input node : ', input_node)
print('output node : ', input_node)

print('----------------------------------------------------------')
print('------------------------DiSNet----------------------------')
print('----------------------------------------------------------')
print("==================All possible routes====================>")

print(all_paths_with_weights(G, input_node, output_node))

selected_path = select_path(G, input_node, output_node)
print('==================>selected path :', selected_path)

current_point_on_path = 0
split_ratio = []
devices = []
trans_rate_forward = [] # take the transrate between node p and p+1 on the path
par_trans_rate = [] # take from the devices mesh graph the average of the par neighbourhood
execution_path = []

for i in range(0, len(max_par_partitions)):

    current_max_par_partitions = max_par_partitions[i]

    ### check here if there is a better neighbourhood <---> and update current_point_on_path
    ### selecting the best of the remaining ones always
    ### add calculation for going to the best neighbours
    p = determine_opt_neighbours(G, selected_path, current_max_par_partitions, current_point_on_path)
    execution_path.append(p)
    current_point_on_path = selected_path.index(p)
    # p = selected_path[current_point_on_path]

    # print(p, temp_current_point_on_path)

    current_neighbours = []
    for n in G.neighbors(p):
        current_neighbours.append([n, G.nodes[n]['weight'], G[p][n]['weight']])
    
    # snicky and dengerous
    if current_point_on_path >=  len(selected_path)-1:
        path_point = current_point_on_path-1
        next_device = selected_path[path_point]
        current_neighbours.append([p, G.nodes[p]['weight'], G[p][next_device]['weight']])
        trans_rate_forward.append(G[p][selected_path[current_point_on_path-1]]['weight'])
    else:
        path_point = current_point_on_path+1
        next_device = selected_path[path_point]
        current_neighbours.append([p, G.nodes[p]['weight'], G[p][next_device]['weight']])
        trans_rate_forward.append(G[p][selected_path[current_point_on_path+1]]['weight'])

    if len(current_neighbours) < current_max_par_partitions:
        current_max_par_partitions = len(current_neighbours)
        horizontal_split_ratio, device, in_throughput = find_split_ratio(current_neighbours)
        split_ratio.append(horizontal_split_ratio)
        par_trans_rate.append(in_throughput)
        devices.append(device)
    else:
        #select a subset
        subset_current_neighbours = select_subset(current_neighbours, current_max_par_partitions)
        horizontal_split_ratio, device, in_throughput = find_split_ratio(subset_current_neighbours)
        split_ratio.append(horizontal_split_ratio)
        par_trans_rate.append(in_throughput)
        devices.append(device)
    
    ### this is the opposite of the a
    # current_point_on_path = current_point_on_path+1
print("==================Determining split ratio==================>")

print('split ratio : ',split_ratio)
print('selected devices : ',devices)

partition_input = []


# max parallelisable partitions in integers [1-4:5, 5-9:4, 10-14:3, 14-18:2]
layer_range = configurations.layer_range

for l in range(0,18):

    if l < 4:
        point = 0
    elif l < 9 & l >= 4:
        point = 1
    elif l < 14 & l >= 10:
        point = 2
    else:
        point = 3
    # cut = 1
    partition = get_partiton_info_DiSNet(l,l,split_ratio[point]) #how to split each layer
    partition_input.append(partition)

# rendom tests here
print("=====Calculating partitions on the neural network DAG======>")
print(partition_input)
print("===================Inference operations====================>")

comp_rate = split_ratio

for t in range(0, 2):
    #blah blah blah
    # if t == 0:
    #     comp_rate = split_ratio
    # else:
    #     comp_rate = [[3,3,3,3,3],[3,3,3,3],[3,3,3],[3,3]]
    print("++++++++++++++++++++++++RUN : ", t)

    output = input_batch
    infer_time = []
    trans_time_seq = []
    with torch.no_grad():
        for j in range(0,len(max_par_partitions)):

            # print(partition_input[layer_range[j,0]:layer_range[j,1]])
            output,sub_infer_time = opt_DiSNet(output, layer_range[j], partition_input[layer_range[j,0]:layer_range[j,1]], par_trans_rate[j],comp_rate[j], split_ratio[j], model)
            # print("Output",output.shape)
            # print(probabilities)

            # check if there transfer of neighbourhoods for the trans forward time on path
            fowrd_trans_time = 0
            if j == len(max_par_partitions)-1:
                fowrd_trans_time = 0
            else:
                if execution_path[j] == execution_path[j +1]:
                    if devices[j] == devices[j + 1]:
                        fowrd_trans_time = 0
                    else:
                        fowrd_trans_time = trans_time_forward(output, trans_rate_forward[j]*10,layer_range[j])
                else:
                    fowrd_trans_time = trans_time_forward(output, trans_rate_forward[j],layer_range[j])

                
            # fowrd_trans_time = 0
            trans_time_seq.append(fowrd_trans_time)
            print('trans time forward ', fowrd_trans_time)
            print("sub trans_rate ",trans_rate_forward[j])
            print('sub infer time ', sub_infer_time)
            infer_time.append(sub_infer_time)
            
        print("--------------------------------------------------------")            
        print('End to end inference time ', np.sum(infer_time)+ np.sum(trans_time_seq))
        print("========================================================")

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        with open("opt/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for k in range(top5_prob.size(0)):
            print(categories[top5_catid[k]], top5_prob[k].item()) 

############################################################################################################
print('---------------------------------------------------------')
print('-------------------------MoDNN---------------------------')
print('---------------------------------------------------------')

modnn_devices = []

for m_n in G.neighbors(input_node):
        modnn_devices.append([m_n, G.nodes[m_n]['weight'], G[input_node][m_n]['weight']] )

modnn_devices.append([input_node, G.nodes[input_node]['weight'], 0])

num_devices = len(modnn_devices)

if num_devices > max_par_partitions[2]:
    sub_modnn_devices = modnn_devices[:max_par_partitions[2]]
    num_devices = len(sub_modnn_devices)
    comp_rate_modnn, device_modnn, in_throughput_modnn = find_split_ratio(sub_modnn_devices)

    trans_rate_modnn = in_throughput_modnn

else:
    comp_rate_modnn, device_modnn, in_throughput_modnn = find_split_ratio(modnn_devices)

    trans_rate_modnn = in_throughput_modnn


# inference modnn
for i in range(0, 2):
    partition_input = []
    for m in range(0,18):
        partition = get_partiton_info(m,m,num_devices) #how to split each layer
        partition_input.append(partition)
    #     print(partition)
    # print(partition_input)
    with torch.no_grad():
        for j in range(0,2):
            output,infer_time = opt_modnn(input_batch, partition_input, trans_rate_modnn,comp_rate_modnn, model)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # print(probabilities)
            print("trans_rate ",trans_rate_modnn)
            print("--------------------------------------------------------")
            print('infer time ', infer_time)
            print("--------------------------------------------------------")


            with open("opt/imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            # Show top categories per image
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for k in range(top5_prob.size(0)):
                print(categories[top5_catid[k]], top5_prob[k].item()) 
            print("--------------------------------------------------------")