import math
import time
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

resources = {}
vertices_no = 0


def trans_time_forward(tensor, trans_rate, layer_range):
    
    out_tensor = tensor.size()
    # print(out_tensor)
    if layer_range[1] == 18:
        t_sub_send = 0
    else:  
        t_sub_send = 32*out_tensor[1]*(out_tensor[2])*out_tensor[3]/(1024*1024*trans_rate)

    return t_sub_send

def select_subset(current_neighbours, current_max_par_partitions):
  sorted_current_neighbours = sorted(current_neighbours, key=lambda x: x[1], reverse=True)

  top_neighbours = sorted_current_neighbours[:current_max_par_partitions]

  selected_neighours = []
  #say unsorted
  for i in range(0, len(current_neighbours)):
    if current_neighbours[0][0] == top_neighbours[0][0]:
      selected_neighours.append(current_neighbours[i])
  return top_neighbours

def select_subset_neighbours(neighbours, current_max_par_partitions):
  sorted_current_neighbours = sorted(neighbours, reverse=True)

  top_neighbours = sorted_current_neighbours[:current_max_par_partitions]
  
  return top_neighbours

def find_split_ratio(current_neighbours):

  split_ratio = []
  nodes = []
  trans_rate = []

  for i in range(0, len(current_neighbours)):
    split_ratio.append(current_neighbours[i][1])
    nodes.append(current_neighbours[i][0])
    trans_rate.append(current_neighbours[i][2])
  throughput = sum(trans_rate) / len(trans_rate)

  return split_ratio, nodes, throughput