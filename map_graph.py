import torch
import networkx as nx
import numpy as np

from collections import defaultdict, deque
import random
from typing import Optional, Tuple, Union, cast
from risk_helper.game import Game
from risk_shared.models.card_model import CardModel
from risk_shared.queries.query_attack import QueryAttack
from risk_shared.queries.query_claim_territory import QueryClaimTerritory
from risk_shared.queries.query_defend import QueryDefend
from risk_shared.queries.query_distribute_troops import QueryDistributeTroops
from risk_shared.queries.query_fortify import QueryFortify
from risk_shared.queries.query_place_initial_troop import QueryPlaceInitialTroop
from risk_shared.queries.query_redeem_cards import QueryRedeemCards
from risk_shared.queries.query_troops_after_attack import QueryTroopsAfterAttack
from risk_shared.queries.query_type import QueryType
from risk_shared.records.moves.move_attack import MoveAttack
from risk_shared.records.moves.move_attack_pass import MoveAttackPass
from risk_shared.records.moves.move_claim_territory import MoveClaimTerritory
from risk_shared.records.moves.move_defend import MoveDefend
from risk_shared.records.moves.move_distribute_troops import MoveDistributeTroops
from risk_shared.records.moves.move_fortify import MoveFortify
from risk_shared.records.moves.move_fortify_pass import MoveFortifyPass
from risk_shared.records.moves.move_place_initial_troop import MovePlaceInitialTroop
from risk_shared.records.moves.move_redeem_cards import MoveRedeemCards
from risk_shared.records.moves.move_troops_after_attack import MoveTroopsAfterAttack
from risk_shared.records.record_attack import RecordAttack
from risk_shared.records.types.move_type import MoveType

#This creates the tensor for the current state of the game, providing the current board state
#(territories, troops in the territories, ownership of the territories), the cards in the player's hand, and the current query type

# Define a graph using NetworkX
G = nx.Graph()

game = Game()

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

class CardModel(BaseModel):
    card_id: int
    territory_id: Optional[int]
    symbol: Union[Literal["Infantry"], Literal["Cavalry"], Literal["Artillery"], Literal["Wildcard"]]

def get_card_set(self, cards: list[CardModel]) -> Optional[Tuple[CardModel, CardModel, CardModel]]:
    cards_by_symbol: dict[str, list[CardModel]] = defaultdict(list)
    for card in cards:
        cards_by_symbol[card.symbol].append(card)

    # Try to make a different symbols set.
    symbols_held = [symbol for symbol in cards_by_symbol.keys() if len(cards_by_symbol[symbol]) > 0]
    if len(symbols_held) >= 3:
        return (cards_by_symbol[symbols_held[0]][0], cards_by_symbol[symbols_held[1]][0], cards_by_symbol[symbols_held[2]][0])

    # To prevent implicitly modifying the dictionary in the for loop, we explicitly initialise the "Wildcard" key.
    cards_by_symbol["Wildcard"]

    # Try to make a matching symbols set.
    for symbol, _cards in cards_by_symbol.items():
        if symbol == "Wildcard":
            continue

        if len(_cards) >= 3:
            return (_cards[0], _cards[1], _cards[2])
        elif len(_cards) == 2 and len(cards_by_symbol["Wildcard"]) >= 1:
            return (_cards[0], _cards[1], cards_by_symbol["Wildcard"][0])
        elif len(_cards) == 1 and len(cards_by_symbol["Wildcard"]) >= 2:
            return (_cards[0], cards_by_symbol["Wildcard"][0], cards_by_symbol["Wildcard"][1])

def create_cards() -> dict[int, CardModel]:
    cards = [
        {"card_id": 0, "territory_id": 0, "symbol": "Infantry"},
        {"card_id": 1, "territory_id": 1, "symbol": "Cavalry"},
        {"card_id": 2, "territory_id": 2, "symbol": "Artillery"},
        {"card_id": 3, "territory_id": 3, "symbol": "Artillery"},
        {"card_id": 4, "territory_id": 4, "symbol": "Cavalry"},
        {"card_id": 5, "territory_id": 5, "symbol": "Artillery"},
        {"card_id": 6, "territory_id": 6, "symbol": "Cavalry"},
        {"card_id": 7, "territory_id": 7, "symbol": "Cavalry"},
        {"card_id": 8, "territory_id": 8, "symbol": "Artillery"},
        {"card_id": 9, "territory_id": 9, "symbol": "Artillery"},
        {"card_id": 10, "territory_id": 10, "symbol": "Infantry"},
        {"card_id": 11, "territory_id": 11, "symbol": "Artillery"},
        {"card_id": 12, "territory_id": 12, "symbol": "Cavalry"},
        {"card_id": 13, "territory_id": 13, "symbol": "Artillery"},
        {"card_id": 14, "territory_id": 14, "symbol": "Cavalry"},
        {"card_id": 15, "territory_id": 15, "symbol": "Artillery"},
        {"card_id": 16, "territory_id": 16, "symbol": "Cavalry"},
        {"card_id": 17, "territory_id": 17, "symbol": "Infantry"},
        {"card_id": 18, "territory_id": 18, "symbol": "Cavalry"},
        {"card_id": 19, "territory_id": 19, "symbol": "Cavalry"},
        {"card_id": 20, "territory_id": 20, "symbol": "Artillery"},
        {"card_id": 21, "territory_id": 21, "symbol": "Infantry"},
        {"card_id": 22, "territory_id": 22, "symbol": "Infantry"},
        {"card_id": 23, "territory_id": 23, "symbol": "Infantry"},
        {"card_id": 24, "territory_id": 24, "symbol": "Infantry"},
        {"card_id": 25, "territory_id": 25, "symbol": "Cavalry"},
        {"card_id": 26, "territory_id": 26, "symbol": "Cavalry"},
        {"card_id": 27, "territory_id": 27, "symbol": "Cavalry"},
        {"card_id": 28, "territory_id": 28, "symbol": "Infantry"},
        {"card_id": 29, "territory_id": 29, "symbol": "Artillery"},
        {"card_id": 30, "territory_id": 30, "symbol": "Infantry"},
        {"card_id": 31, "territory_id": 31, "symbol": "Infantry"},
        {"card_id": 32, "territory_id": 32, "symbol": "Infantry"},
        {"card_id": 33, "territory_id": 33, "symbol": "Infantry"},
        {"card_id": 34, "territory_id": 34, "symbol": "Infantry"},
        {"card_id": 35, "territory_id": 35, "symbol": "Cavalry"},
        {"card_id": 36, "territory_id": 36, "symbol": "Cavalry"},
        {"card_id": 37, "territory_id": 37, "symbol": "Artillery"},
        {"card_id": 38, "territory_id": 38, "symbol": "Artillery"},
        {"card_id": 39, "territory_id": 39, "symbol": "Infantry"},
        {"card_id": 40, "territory_id": 40, "symbol": "Artillery"},
        {"card_id": 41, "territory_id": 41, "symbol": "Artillery"},
        {"card_id": 42, "territory_id": None, "symbol": "Wildcard"},
        {"card_id": 43, "territory_id": None, "symbol": "Wildcard"}
    ]

    cards = dict([(card["card_id"], CardModel(**card)) for card in cards])
    return cards

#card ids range from 0 - 43 (one for each territory and 2 wildcards)

"""
def get_all_adjacent_territories(self, territories: list[int]) -> list[int]:
    result = []
    for territory in territories:
        result.extend(self.map.get_adjacent_to(territory))

    return list(set(result) - set(territories))
"""

"""
class CardModel(BaseModel):
    card_id: int
    territory_id: Optional[int]
    symbol: Union[Literal["Infantry"], Literal["Cavalry"], Literal["Artillery"], Literal["Wildcard"]]
"""

"""
class TerritoryModel(BaseModel):
    territory_id: int
    occupier: Optional[int]
    troops: int
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

def form_state(graph, nodes, cards):
    for node in nodes:
        graph.add_node(node.id, weight=node.troops, owner=node.occupier)
    for node in nodes:
        for neighbor in game.state.map.get_adjacent_to(node):
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

    #Create a matrix of card
    cards_matrix = convert_card_sets_to_matrix(cards)

    # Add the card matrix as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, cards_matrix))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

def update_state(graph, nodes, cards):
    for node in nodes:
        graph.nodes[node.id]['weight'] = node.troops
        graph.nodes[node.id]['owner'] = node.occupier

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

    #Create a matrix of card
    cards_matrix = convert_card_sets_to_matrix(cards)

    # Add the card matrix as a new layer in the feature matrix
    feature_matrix = np.dstack((feature_matrix, cards_matrix))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

"""
def form_state(graph, nodes, cards):
    # Add nodes and edges to the graph
    for node in nodes:
        graph.add_node(node.id, weight=node.troops, owner=node.occupier)
    for node in nodes:
        for neighbor in game.state.map.get_adjacent_to(node):
            graph.add_edge(node, neighbor)

    # Create an adjacency matrix with node features
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    np.fill_diagonal(adjacency_matrix, [data['weight'] for node, data in graph.nodes(data=True)])

    # Create an ownership matrix
    num_players = 5
    ownership_matrix = np.zeros((len(graph.nodes), num_players))
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        ownership_matrix[i, data['owner']] = 1

    # Create a matrix for card information
    cards_matrix = convert_card_sets_to_matrix(cards)

    # Stack the adjacency matrix, ownership matrix, and card matrix along the third dimension
    feature_matrix = np.dstack((adjacency_matrix, ownership_matrix, cards_matrix))

    # Convert the feature matrix to a tensor
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    return feature_tensor
"""
