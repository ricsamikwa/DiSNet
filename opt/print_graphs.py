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

mesh_network_id = 3 #reserve 0 - 3

## generate mesh network graph with num_devices devices and num_connections connections with random resources and transmission throughput
# print('Generating random network graph of heterogenous resources (comp, network)')
# G = generate_random_graph(num_devices, num_connections)

# draw_graph(G, mesh_network_id) # picture

# save_graph(G, mesh_network_id) # dump the backup

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## reading existing graph

print('Loading existing network graph of heterogenous resources (comp, network)')
G = read_graph(mesh_network_id)

draw_graph_pdf(G, mesh_network_id) 
