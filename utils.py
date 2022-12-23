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
        t_sub_send = 32*out_tensor[1]*(out_tensor[2])*out_tensor[3]/(1024*1024*1024*trans_rate)

    return t_sub_send

# Add a vertex to the dictionary
def add_vertex(v):
  global resources
  global vertices_no
  if v in resources:
    print("Vertex ", v, " already exists.")
  else:
    vertices_no = vertices_no + 1
    resources[v] = []

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
  global resources
  # Check if vertex v1 is a valid vertex
  if v1 not in resources:
    print("Vertex ", v1, " does not exist.")
  # Check if vertex v2 is a valid vertex
  elif v2 not in resources:
    print("Vertex ", v2, " does not exist.")
  elif v2 == v1:
    print("Same node")
  else:
    # Since this code is not restricted to a directed or 
    # an undirected graph, an edge between v1 v2 does not
    # imply that an edge exists between v2 and v1
    temp = [v2, e]
    resources[v1].append(temp)

# Print the graph
def print_graph():
  global resources
  for vertex in resources:
    for edges in resources[vertex]:
      print(vertex, " -> ", edges[0], " edge weight: ", edges[1])

def create_resources_graph(num_devices, list_comp_rates, list_trans_rate):
    # stores the number of vertices in the grap
    global resources
    comp_rate = []
    device_keys = []

    for i in range(0,num_devices):
        add_vertex(i+1)
        comp_rate.append(random.choice(list_comp_rates))
        device_keys.append(i+1)

    # for i in range(0,num_devices*2):
    #     add_edge(random.choice(device_keys), random.choice(device_keys), random.choice(list_trans_rate))
    add_edge(1, 2, random.choice(list_trans_rate))
    add_edge(1, 3, random.choice(list_trans_rate))
    add_edge(2, 3, random.choice(list_trans_rate))
    add_edge(3, 4, random.choice(list_trans_rate))
    add_edge(4, 1, random.choice(list_trans_rate))
    print_graph()
    # Reminder: the second element of each list inside the dictionary
    # denotes the edge weight.
    print ("Internal representation: ", resources)

    return resources, comp_rate


# def find_all_paths(start, end, path=[]):
#     global graph

#     path = path + [start]
#     if start == end:
#         return [path]
#     if start not in graph:
#         return []
#     paths = []
#     for node in graph[start]:
#         if node not in path:
#             newpaths = find_all_paths(node, end, path)
#             for newpath in newpaths:
#                 paths.append(newpath)
#     return paths

# def find_shortest_path(start, end, path=[]):
#     global graph

#     path = path + [start]
#     if start == end:
#         return path
#     if start not in graph:
#         return None
#     shortest = None
#     for node in graph[start]:
#         if node not in path:
#             newpath = find_shortest_path( node, end, path)
#             if newpath:
#                 if not shortest or len(newpath) < len(shortest):
#                     shortest = newpath
#     return shortest

 # driver code

# paths = find_all_paths(1, 4)
# print(paths)
# path = find_shortest_path(1, 4)
# print(path)
# def read_graph():
#     graph_adjacency_list = { }
#     for line in open("input.txt"):
#         line = map(int, line.rstrip("\t\r\n").split("\t"))
#         graph_adjacency_list.update({ line[0]: { e: 1 for e in line[1:] } })

#     return graph_adjacency_list

# # graph_data = read_graph()
# G = nx.Graph(graph)
# nx.draw_networkx(G, with_labels = True, node_color = "c", edge_color = "k", font_size = 8)

# plt.axis('off')
# plt.draw()
# plt.savefig("graph.pdf")