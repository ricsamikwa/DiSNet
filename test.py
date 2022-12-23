import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq


def generate_random_graph(num_vertices, min_weight, max_weight):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Generate random weights for the vertices
    vertex_weights = {i: random.uniform(min_weight, max_weight) for i in range(num_vertices)}

    # Add the vertices and their weights to the graph
    for vertex, weight in vertex_weights.items():
        G.add_node(vertex, weight=weight)

    # Generate random edges between the vertices
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if random.random() < 0.5:
                # Generate a random weight for the edge
                weight = random.uniform(min_weight, max_weight)
                # Add the edge and its weight to the graph
                G.add_edge(i, j, weight=weight)

    return G

# Generate a random weighted graph with 5 vertices and weights between 0 and 1
G = generate_random_graph(5, 0, 1)

# Print the graph
print(G.edges(data=True))
print(nx.get_node_attributes(G, 'weight'))
# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)

# Label the edges with their weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Label the vertices with their weights
vertex_labels = nx.get_node_attributes(G, 'weight')
nx.draw_networkx_labels(G, pos, labels=vertex_labels)

plt.show()


#creating the graph

# Define a dictionary to store the vertices and their weights
vertices = {
    'A': 3,
    'B': 5,
    'C': 2,
    'D': 4,
    'E': 1
}

# Define a dictionary to store the edges and their weights
edges = {
    ('A', 'B'): 3,
    ('A', 'C'): 1,
    ('B', 'C'): 2,
    ('B', 'D'): 4,
    ('C', 'D'): 1,
    ('D', 'E'): 3,
}

# Print the graph
print(vertices)
print(edges)


# find all paths between to vertices in the graph
def dfs_paths(vertices, edges, start, end, path=[]):
    # Add the start vertex to the path
    path = path + [start]
    # If the start and end vertices are the same, return the path
    if start == end:
        return [path]
    # Initialize an empty list to store the paths
    paths = []
    # For each neighbor of the start vertex
    for neighbor, weight in edges.items():
        if neighbor[0] == start:
            # If the neighbor has not been visited yet
            if neighbor[1] not in path:
                # Find the paths from the neighbor to the end vertex
                new_paths = dfs_paths(vertices, edges, neighbor[1], end, path)
                # Add the paths to the list
                for new_path in new_paths:
                    paths.append(new_path)
    return paths

# Define the start and end vertices
start = 'A'
end = 'E'

# Find all paths from the start vertex to the end vertex
paths = dfs_paths(vertices, edges, start, end)

# Print the paths
print(f"All paths from {start} to {end}:")
for path in paths:
    print(path)

#longest path

def dijkstra_longest_path(vertices, edges, start):
    # Create a dictionary to store the distances from the start vertex
    distances = {vertex: float('-inf') for vertex in vertices}
    distances[start] = 0

    # Create a priority queue to store the vertices that have been visited
    queue = []
    heapq.heappush(queue, [0, start])

    # Keep track of the previous vertex for each vertex
    previous = {vertex: None for vertex in vertices}

    # While there are vertices in the queue
    while queue:
        # Pop the vertex with the longest distance from the queue
        current_distance, current_vertex = heapq.heappop(queue)

        # For each neighbor of the current vertex
        for neighbor, weight in edges.items():
            if neighbor[0] == current_vertex:
                # Calculate the distance to the neighbor
                distance = current_distance + weight + vertices[neighbor[1]]
                # If the distance to the neighbor is longer than the current distance
                if distance > distances[neighbor[1]]:
                    # Update the distance to the neighbor
                    distances[neighbor[1]] = distance
                    # Update the previous vertex for the neighbor
                    previous[neighbor[1]] = current_vertex
                    # Add the neighbor to the queue
                    heapq.heappush(queue, [distance, neighbor[1]])

    return distances, previous

# Define the start and end vertices
start = 'A'
end = 'E'

# Find the longest path from the start vertex to the end vertex
distances, previous = dijkstra_longest_path(vertices, edges, start)

# Print the longest distance
print(f"Longest distance from {start} to {end}: {distances[end]}")

# Initialize the path with the end vertex
path = [end]

# Get the previous vertex for the end vertex
previous_vertex = previous[end]

# While there is a previous vertex
while previous_vertex is not None:
    # Add the previous vertex to the path
    path.append(previous_vertex)
    # Update the previous vertex
    previous_vertex = previous[previous_vertex]

# Reverse the path
path = path[::-1]

# Print the longest path
print(f"Longest path from {start} to {end}: {path}")



#shortest path
def dijkstra(vertices, edges, start):
    # Create a dictionary to store the distances from the start vertex
    distances = {vertex: float('inf') for vertex in vertices}
    distances[start] = 0

    # Create a priority queue to store the vertices that have been visited
    queue = []
    heapq.heappush(queue, [0, start])

    # Keep track of the previous vertex for each vertex
    previous = {vertex: None for vertex in vertices}

    # While there are vertices in the queue
    while queue:
        # Pop the vertex with the shortest distance from the queue
        current_distance, current_vertex = heapq.heappop(queue)

        # For each neighbor of the current vertex
        for neighbor, weight in edges.items():
            if neighbor[0] == current_vertex:
                # Calculate the distance to the neighbor
                distance = current_distance + weight + vertices[neighbor[1]]
                # If the distance to the neighbor is shorter than the current distance
                if distance < distances[neighbor[1]]:
                    # Update the distance to the neighbor
                    distances[neighbor[1]] = distance
                    # Update the previous vertex for the neighbor
                    previous[neighbor[1]] = current_vertex
                    # Add the neighbor to the queue
                    heapq.heappush(queue, [distance, neighbor[1]])

    return distances, previous

# Define the start and end vertices
start = 'A'
end = 'E'

# Find the shortest path from the start vertex to the end vertex
distances, previous = dijkstra(vertices, edges, start)

# Print the shortest distance
print(f"Shortest distance from {start} to {end}: {distances[end]}")

# Initialize the path with the end vertex
path = [end]

# Get the previous vertex for the end vertex
previous_vertex = previous[end]

# While there is a previous vertex
while previous_vertex is not None:
    # Add the previous vertex to the path
    path.append(previous_vertex)
    # Update the previous vertex
    previous_vertex = previous[previous_vertex]

# Reverse the path
path = path[::-1]

# Print the shortest path
print(f"Shortest path from {start} to {end}: {path}")
