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
import csv



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

model = VGG16()
model_dict = model.state_dict()
model_dict.update(torch.load("main/vgg16-modify.pth"))
model.load_state_dict(model_dict)
model.eval()
device_pace_rate = configurations.DEVICE_PACE_RATE 


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda:0')
    model.to('cuda:0')


mesh_network_id = 0 # pos 0 - 3
num_runs = 1
num_devices = 10
num_connections = 15

save_to_file = False
run_num = 1

def run_test(max_bandwidth):
    # energy flags
    energy_sensitivity = 0.5 

    ## generate mesh network graph with num_devices devices and num_connections connections with random resources and transmission throughput
    # print('Generating random network graph of heterogenous resources (comp, network)')
    # G = generate_random_graph(num_devices, num_connections)

    # draw_graph(G, mesh_network_id) # picture

    # save_graph(G, mesh_network_id) # dump the backup
        

    ## reading existing graph
    print('Loading existing network graph of heterogenous resources (comp, network)')
    G = read_graph(mesh_network_id)

    max_par_partitions = configurations.max_par_partitions 

    # random input and output node pairs
    while True:
        input_node = random.randint(0, num_devices -1)
        output_node = random.randint(0, num_devices -1)
        if input_node != output_node:
            break

    # use previous ones 
    input_node = 8
    output_node = 4

    print('input node : ', input_node)
    print('output node : ', output_node)

    print('------------------------DiSNet----------------------------')
    print("All possible paths :")

    print(all_paths_with_weights(G, input_node, output_node))



    selected_path = select_path(G, input_node, output_node, energy_sensitivity)

    print('selected path :', selected_path)

    current_point_on_path = 0
    split_ratio = []
    devices = []
    trans_rate_forward = [] # transimission rate node p and p+1
    par_trans_rate = [] # neighbourhood average from devices mesh graph
    execution_path = []

    for i in range(0, len(max_par_partitions)):

        current_max_par_partitions = max_par_partitions[i]

        ### check here if there is a better neighbourhood <---> and update current
    
        p = determine_opt_neighbours(G, selected_path, current_max_par_partitions, current_point_on_path, energy_sensitivity)
        execution_path.append(p)
        current_point_on_path = selected_path.index(p)
       
        current_neighbours = []
        for n in G.neighbors(p):
            current_neighbours.append([n, G.nodes[n]['weight'], G[p][n]['weight']])
        
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
        

    print('split ratio : ',split_ratio)
    print('selected devices : ',devices)

   
    trans_rate_forward = cap_trans_rate(max_bandwidth,trans_rate_forward)
    par_trans_rate = cap_trans_rate(max_bandwidth,par_trans_rate)

    layer_range = configurations.layer_range

    partition_input = create_partition_input(split_ratio)

    print("Partitions on the DAG======>")
    print(partition_input)
    print("Inference operations=======>")

    comp_rate = split_ratio


    for t in range(0, num_runs):
        
        infer_time = 0
        infer_accurancy = 0
        infer_energy = 0

        if t == 0:
            print("RUN : ", t)

        output = input_batch
        infer_time = []
        trans_time_seq = []
        with torch.no_grad():
            for j in range(0,len(max_par_partitions)):

                # print(partition_input[layer_range[j,0]:layer_range[j,1]])
                output,sub_infer_time = opt_DiSNet(output, layer_range[j], partition_input[layer_range[j,0]:layer_range[j,1]], par_trans_rate[j],comp_rate[j], split_ratio[j], model)
              
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

                    
                fowrd_trans_time = fowrd_trans_time/device_pace_rate

                infer_energy += partition_energy(devices[j],sub_infer_time,fowrd_trans_time)
                
                trans_time_seq.append(fowrd_trans_time)
                
                infer_time.append(sub_infer_time)

            infer_time = np.sum(infer_time)+ np.sum(trans_time_seq)

            if t == 0:
                           
                print('End to end inference time ', infer_time)
                print('energy consumption ', infer_energy)
                print("===================================================")

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            with open("main/imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            
            # top categories
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            if t == 0:
                for k in range(top5_prob.size(0)):
                    print(categories[top5_catid[k]], top5_prob[k].item()) 

            infer_accurancy = top5_prob[0].item()
            top5_accurancy = sum(top5_prob).item()
            if t == 0:
                print('top 5 acc: ',sum(top5_prob).item())
       


    print('-------------------------Modnn---------------------------')


    modnn_devices = []

    for m_n in G.neighbors(input_node):
            modnn_devices.append([m_n, G.nodes[m_n]['weight'], G[input_node][m_n]['weight']] )

    modnn_devices.append([input_node, G.nodes[input_node]['weight'], 0])

    num_devices_modnn = len(modnn_devices)

    num_splits = max_par_partitions[0]

    if num_devices_modnn > num_splits:
        sub_modnn_devices = modnn_devices[:num_splits]
        num_devices_modnn = len(sub_modnn_devices)
        comp_rate_modnn, device_modnn, in_throughput_modnn = find_split_ratio(sub_modnn_devices)

        trans_rate_modnn = in_throughput_modnn

    else:
        comp_rate_modnn, device_modnn, in_throughput_modnn = find_split_ratio(modnn_devices)

        trans_rate_modnn = in_throughput_modnn

    if trans_rate_modnn > max_bandwidth:
        trans_rate_modnn = max_bandwidth

    print('comp_rate_modnn ',comp_rate_modnn)
    print('trans_rate_modnn ',trans_rate_modnn)


    partition_input = []
    for m in range(0,18):
        if m > 12:
            if num_devices_modnn > max_par_partitions[3]:
                num_devices_modnn = max_par_partitions[3]
        partition = get_partiton_info(m,m,num_devices_modnn) 
        partition_input.append(partition)
    #     print(partition)
    # print(partition_input)

    for i in range(0, num_runs):
        infer_time_modnn = 0
        infer_energy = 0
        
        with torch.no_grad():

            output,infer_time_modnn,sub_trans_time = opt_modnn(input_batch, partition_input, trans_rate_modnn,comp_rate_modnn, model)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # print(probabilities)
            
            infer_energy = partition_energy(device_modnn,infer_time_modnn,sub_trans_time)

            
            if i == 0:
                print("--------------------------------------------------------")
                print('infer time ', infer_time_modnn)
                print('energy consumption ', infer_energy)
                print("--------------------------------------------------------")


            with open("main/imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            # top categories
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            if i == 0:
                for k in range(top5_prob.size(0)):
                    print(categories[top5_catid[k]], top5_prob[k].item()) 
                print("--------------------------------------------------------")
            
            if i == 0:
                print('top 5 acc: ',sum(top5_prob).item())
      

throughput_cap = get_throughput_cap()

for i in throughput_cap:
    print("Test: ",run_num)
    print("B Max: ",i)
    run_test(i)

  