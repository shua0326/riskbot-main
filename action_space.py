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
                num_territories = 42
                actions_tensor = torch.arange(num_territories).int().unsqueeze(0)
                return actions_tensor

            case "PlaceInitialTroop":
                num_territories = 42
                actions_tensor = torch.arange(num_territories).int().unsqueeze(0)
                return actions_tensor

            case "RedeemCards":
                num_cards = 45      #number 45 means to pass the turn
                actions_tensor = torch.arange(num_cards).int().unsqueeze(0)
                return actions_tensor

            case "DistributeTroops":
                #unsure how to implement as this is very complicated, returns a dictionary
                #which contains the decided distribution of all redeemed troops
                #{territory:number_of_troops}
                num_territories = 42
                total_troops = 50       #arbitrary max number of troops that can be obtained through redeeming

                # Generate combinations of territories and troop counts
                actions = []
                for i in range(num_territories):
                    for j in range(1, total_troops + 1):
                        actions.append((i, j))

                actions_tensor = torch.tensor(actions).int()
                return actions_tensor

            case "Attack":
                num_my_territories = 42
                num_enemy_territories = 42
                num_attacking_troops = 3

                actions = []
                for i in range(num_my_territories):
                    for j in range(num_enemy_territories):
                        for troops in range(1, 4):  # Attack with 1, 2, or 3 troops
                            actions.append((i, j, troops))

                actions.append((-1, -1, -1))        #this result means to pass the turn

                actions_tensor = torch.tensor(actions).int()
                return actions_tensor

            case "TroopsAfterAttack":
                num_troops_to_move = 20     #arbitrary max number of troops that could be on a territory
                actions_tensor = torch.arange(num_troops_to_move).int().unsqueeze(0)
                return actions_tensor

            case "Defend":
                num_troops_to_defend_with = 2
                actions_tensor = torch.arange(num_troops_to_defend_with).int().unsqueeze(0)
                return actions_tensor

            case "Fortify":
                num_source_territories = 42
                num_target_territories = 42
                max_troops = 20     #arbitrary number for max number of troops on a territory

                actions = []
                for i in range(num_source_territories):
                    for j in range(num_target_territories):
                        for troops in range(1, max_troops):  # Attack with 1, 2, or 3 troops
                            actions.append((i, j, troops))

                actions.append((-1, -1, -1))        #this result means to pass the turn

                actions_tensor = torch.tensor(actions).int()
                return actions_tensor

        """from_territory = np.random.choice(len(self.my_territories))
        to_territory = np.random.choice(len(self.adjacent_territories))
        num_troops = np.random.randint(1, self.max_troops + 1)
        return from_territory, to_territory, num_troops"""