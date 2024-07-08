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

card_symbol_mapping = {"Infantry": 0, "Cavalry": 1, "Artillery": 2, "Wildcard": 3}

#card ids range from 0 - 43 (one for each territory and 2 wildcards)
"""
class CardModel(BaseModel):
    card_id: int
    territory_id: Optional[int]
    symbol: Union[Literal["Infantry"], Literal["Cavalry"], Literal["Artillery"], Literal["Wildcard"]]
"""

def convert_card_sets_to_matrix(card_sets, max_card_id=43):
    num_symbols = 4  # Number of different symbols
    # Initialize a zero matrix with an additional dimension for symbols
    card_matrix = np.zeros((len(card_sets), max_card_id, num_symbols))

    for i, card_set in enumerate(card_sets):
        for card in card_set:
            card_id = card.card_id
            symbol_index = card_symbol_mapping[card.symbol]
            # Mark the presence of card ID and symbol using one-hot encoding
            card_matrix[i, card_id, symbol_index] = 1

    return card_matrix

def form_state(graph, nodes, cards, query_type):
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

    #Create a matrix of card sets
    cards_matrix = convert_card_sets_to_matrix(cards)

    # Add the card matrix as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, cards_matrix))

    # Create a one-hot encoding for the query type
    query_type_vector = np.zeros(len(query_type_mapping))
    query_type_vector[query_type_mapping[query_type]] = 1

    # Add the query type vector as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, query_type_vector))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

def update_state(graph, nodes, cards, query_type):
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

    #Create a matrix of card sets
    cards_matrix = convert_card_sets_to_matrix(cards)

    # Add the card matrix as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, cards_matrix))

    # Create a one-hot encoding for the query type
    query_type_vector = np.zeros(len(query_type_mapping))
    query_type_vector[query_type_mapping[query_type]] = 1

    # Add the query type vector as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, query_type_vector))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

