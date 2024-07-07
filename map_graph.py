import torch
import networkx as nx
import numpy as np

# Define a graph using NetworkX
G = nx.Graph()

query_type_mapping = {
    "ClaimTerritory": 0,
    "PlaceInitialTroop": 1,
    "RedeemCards": 2,
    "DistributeTroops": 3,
    "Attack": 4,
    "TroopsAfterAttack": 5,
    "Defend": 6,
    "Fortify": 7
}

def form_graph(graph, nodes, query_type):
    for node in nodes:
        graph.add_node(node, weight=nodes[node]['weight'], owner=nodes[node]['owner'])
    for node in nodes:
        for neighbor in nodes[node]['neighbors']:
            graph.add_edge(node, neighbor)

    # Create an adjacency matrix with node features
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    np.fill_diagonal(adjacency_matrix, [data['weight'] for node, data in graph.nodes(data=True)])

    # Create an ownership matrix
    num_players = 5
    ownership_matrix = np.zeros((len(graph.nodes), num_players))
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        ownership_matrix[i, data['owner']] = 1

    # Stack the adjacency matrix and the ownership matrix along the third dimension
    feature_matrix = np.dstack((adjacency_matrix, ownership_matrix))

    # Create a one-hot encoding for the query type
    query_type_vector = np.zeros(len(query_type_mapping))
    query_type_vector[query_type_mapping[query_type]] = 1

    # Add the query type vector as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, query_type_vector))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

def update_graph(graph, nodes, query_type):
    for node in nodes:
        graph.nodes[node]['weight'] = nodes[node]['weight']
        graph.nodes[node]['owner'] = nodes[node]['owner']

    # Create an adjacency matrix with node features
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    np.fill_diagonal(adjacency_matrix, [data['weight'] for node, data in graph.nodes(data=True)])

    # Create an ownership matrix
    num_players = 5
    ownership_matrix = np.zeros((len(graph.nodes), num_players))
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        ownership_matrix[i, data['owner']] = 1

    # Stack the adjacency matrix and the ownership matrix along the third dimension
    feature_matrix = np.dstack((adjacency_matrix, ownership_matrix))

    # Create a one-hot encoding for the query type
    query_type_vector = np.zeros(len(query_type_mapping))
    query_type_vector[query_type_mapping[query_type]] = 1

    # Add the query type vector as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, query_type_vector))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

