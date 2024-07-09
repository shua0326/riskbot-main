import numpy as np
import torch

"""
match query:
    case QueryClaimTerritory() as 1:

    case QueryPlaceInitialTroop() as 2:

    case QueryRedeemCards() as 3:

    case QueryDistributeTroops() as 4:

    case QueryAttack() as 5:

    case QueryTroopsAfterAttack() as 6:

    case QueryDefend() as 7:

    case QueryFortify() as 8:
"""

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

class ActionSpace:
    def __init__(self):
        self.query = ""
        self.my_territories = []    #potentially have this as a tuple of (territory, troops, list of adjacent territories)
        self.adjacent_territories = []
        self.cards = []
        self.total_troops = 0

    def update(self, query, my_territories, adjacent_territories, cards, total_troops):
        self.query = query
        self.my_territories = my_territories
        self.adjacent_territories = adjacent_territories
        self.cards = cards
        self.total_troops = total_troops

    #need to change this so that a list/tensor of actions is constructed first
    def create_action_space(self):
        match self.query:
            case "ClaimTerritory":
                num_query_types = 8
                num_territories = 42
                actions_tensor = torch.zeros((num_query_types, num_territories))

                # Encode the query type
                query_type_index = query_type_mapping[self.query]
                for action in self.adjacent_territories:
                    # One-hot encode the territory ID
                    actions_tensor[query_type_index, action.id] = 1

                actions_tensor = actions_tensor.unsqueeze(0).float()

                return actions_tensor

            case "PlaceInitialTroop":
                num_query_types = len(query_type_mapping)
                num_territories = 42
                max_troops = self.total_troops
                # Initialize the tensor with an extra dimension for troop counts
                actions_tensor = torch.zeros((num_query_types, num_territories, max_troops))

                query_type_index = query_type_mapping[self.query]
                for territory in self.my_territories:
                    for troop_count in range(1, max_troops + 1):
                        # Encode the possibility of placing each number of troops in each territory
                        actions_tensor[query_type_index, territory.id, troop_count - 1] = 1

                actions_tensor = actions_tensor.unsqueeze(0).float()

                return actions_tensor

            case "RedeemCards":
                return self.redeem_cards()
            case "DistributeTroops":
                return self.distribute_troops()
            case "Attack":
                return self.attack()
            case "TroopsAfterAttack":
                return self.troops_after_attack()
            case "Defend":
                return self.defend()
            case "Fortify":
                return self.fortify()

        """from_territory = np.random.choice(len(self.my_territories))
        to_territory = np.random.choice(len(self.adjacent_territories))
        num_troops = np.random.randint(1, self.max_troops + 1)
        return from_territory, to_territory, num_troops"""