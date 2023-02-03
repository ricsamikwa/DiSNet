import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle
import numpy as np
import configurations
from utils import select_subset_neighbours

trans_powers = configurations.trans_powers
comp_powers = configurations.comp_powers
device_power_groups = configurations.device_power_groups

norm_comp_powers= comp_powers / np.linalg.norm(comp_powers)
norm_trans_powers= trans_powers / np.linalg.norm(trans_powers)

def generate_random_graph(n, m, weight_range=(10,50), comp_rate_range=(1,10)):
    # Create an empty graph
    G = nx.Graph()
    
    # Add n nodes to the graph
    for i in range(n):
        G.add_node(i)
        G.nodes[i]['weight'] = random.randint(*comp_rate_range)
    
    while not nx.is_connected(G):
        u, v = random.sample(range(n), 2)
        if u != v:
            w = random.randint(*weight_range)
            G.add_edge(u, v, weight=w)
    
    #Add more edges but avoid self-loop
    while nx.number_of_edges(G) < m:
        u, v = random.sample(range(n), 2)
        if u != v and not G.has_edge(u, v):
            w = random.randint(*weight_range)
            G.add_edge(u, v, weight=w)
    
    return G

def save_graph(G, mesh_network_id):

    # pickle.dump(G, open(graph_name, 'wb'))
    serialized = pickle.dumps(G)
    graph_name = 'networks/network'+str(mesh_network_id)+'.txt'

    with open(graph_name,'wb') as file_object:
        file_object.write(serialized)

def read_graph(mesh_network_id):

    graph_name = 'networks/network'+str(mesh_network_id)+'.txt'
    # G = pickle.load(open(graph_name, 'rb'))
    with open(graph_name,'rb') as file_object:
        raw_data = file_object.read()

    G = pickle.loads(raw_data)
    return G

def all_paths_with_weights(G, start, end):
    paths = nx.all_simple_paths(G, start, end)
    # longest_path = 0
    total_weights = []
    for path in paths:
        total_weight = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_weight += G[u][v]['weight']
            total_weight += G.nodes[u]['weight']
        total_weights.append((path, total_weight))

    return total_weights

def longest_path(G, start, end):
    if nx.is_directed_acyclic_graph(G):
        return nx.dag_longest_path(G, weight='weight')
    else:
        _, _, pred = nx.bellman_ford_predecessor_and_distance(G, start, weight='weight')
        if end not in pred:
            return None
        longest = [end]
        cur = end
        while cur != start:
            cur = pred[cur]
            longest.append(cur)
        return longest[::-1]
def shortest_path(G, start, end):
    try:
        return nx.shortest_path(G, start, end, weight='weight')
    except nx.NetworkXNoPath:
        return None

def draw_graph(G, mesh_network_id):
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='blue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['weight'] for u, v in G.edges()})
    nx.draw_networkx_labels(G, pos, labels={i: f"{i} ({G.nodes[i]['weight']})" for i in G.nodes()})

    graph_name = 'networks/network'+str(mesh_network_id)+'.png'
    plt.savefig(graph_name)
    # plt.show()
def power_details(node_num):
    comp_power = 0
    trans_power = 0

    # if node_num <= device_power_groups[0]:
    #     comp_power = norm_comp_powers[0]
    #     trans_power = norm_trans_powers[0]
    # elif node_num > device_power_groups[0] & node_num <= device_power_groups[1]:
    #     comp_power = norm_comp_powers[1]
    #     trans_power = norm_trans_powers[1]
    # else:
    #     comp_power = norm_comp_powers[2]
    #     trans_power = norm_trans_powers[2]
    # print(comp_powers)
    if node_num <= device_power_groups[0]:
        comp_power = norm_comp_powers[0]
        trans_power = norm_trans_powers[0]
    elif node_num > device_power_groups[0] & node_num <= device_power_groups[1]:
        comp_power = norm_comp_powers[1]
        trans_power = norm_trans_powers[1]
    else:
        comp_power = norm_comp_powers[2]
        trans_power = norm_trans_powers[2]
    
    return comp_power, trans_power

def select_path(G, input_node, output_node, energy_sensitivity):
    paths = nx.all_simple_paths(G, input_node, output_node)
    selected_path = []
    current_path_weight = 0
    for path in paths:
        total_weight = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            #considering both rate and transmission
            #lets take the highest avarage 
            comp_power, trans_power = power_details(u)
            #for energy the lower the power the better !!
            total_weight += (G[u][v]['weight']/5) * (1 - ((1 - energy_sensitivity)*trans_power)) # the number 5 reduces the influence of the transmission in the selection
            total_weight += G.nodes[u]['weight'] * (1 - ((1 - energy_sensitivity)*comp_power))

        #amortized cost of the path
        # print(total_weight, len(path))
        total_weight = total_weight/len(path)
        # print(total_weight)

        if total_weight > current_path_weight:
            selected_path.append(path)
            current_path_weight = total_weight
        
    return selected_path[len(selected_path)-1]

def determine_opt_neighbours(G, selected_path, current_max_par_partitions, current_point_on_path, energy_sensitivity):
    selected_point = current_point_on_path
    selected_neighbours_weight = 0

    for p in selected_path[current_point_on_path:]:
        total_weight = 0

        neighbours = []
        for n in G.neighbors(p):

            comp_power, trans_power = power_details(n)

            neighbours.append(G.nodes[n]['weight'] * (1-((1 - energy_sensitivity)*comp_power)))
            # total_weight += G[n][p]['weight']/5 + G.nodes[n]['weight']
        
        # was n before !!
        # neighbours.append(G.nodes[n]['weight'])
        comp_power, trans_power = power_details(p)
        neighbours.append(G.nodes[n]['weight'] * (1-((1 - energy_sensitivity)*comp_power)))

        # here!
        #check weights later

        if len(neighbours) > current_max_par_partitions:
            # print("neighbours :", neighbours)
            subset_neighbors = select_subset_neighbours(neighbours, current_max_par_partitions)
            # print("subset_neighbors :", subset_neighbors)

            total_weight = sum(subset_neighbors)/current_max_par_partitions
        else:
            total_weight = sum(neighbours)/ len(neighbours)

        if p == selected_point:
            selected_neighbours_weight = total_weight
        if selected_neighbours_weight < total_weight:
            selected_point = p
            selected_neighbours_weight = total_weight

    return selected_point

def partition_energy(devices,sub_infer_time,fowrd_trans_time):

    partition_energy = 0
    for device in devices:
        comp_power, trans_power = power_details(device)
        partition_energy += sub_infer_time*comp_power + fowrd_trans_time*trans_power

    return partition_energy

def partition_energy_modified(device_modnn,infer_time_modnn,sub_trans_time):

    partition_energy = 0
    sub_infer_time = infer_time_modnn - sub_trans_time
    for device in device_modnn:
        comp_power, trans_power = power_details(device)
        partition_energy += sub_infer_time*comp_power + sub_trans_time*trans_power

    return partition_energy

# generate a random graph with 10 nodes and 15 edgespy
# G = generate_random_graph(10, 15)
# # x = [[G.nodes[n]['weight'] ,G[2][n]['weight'] ] for n in G.neighbors(2)]
# # print(x)
# draw_graph(G)
# print(all_paths_with_weights(G, 0, 9))
# print(select_path(G, 0, 9))
# print(shortest_path(G, 0, 9))


