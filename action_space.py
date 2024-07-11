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
                cards = game.state.cards
                all_combinations = combinations(cards.values(), 3)

                # Filter valid sets
                valid_sets = [combo for combo in all_combinations if game.state.get_card_set(combo) is not None]

                valid_sets_ids = [[cards[i].card_id for i in combo] for combo in valid_sets]

                valid_sets_ids.append([-1, -1, -1])

                actions_tensor = torch.tensor(valid_sets_ids).int()
                return actions_tensor

            case "DistributeTroops":
                #unsure how to implement as this is very complicated, returns a dictionary
                #which contains the decided distribution of all redeemed troops
                #{territory:number_of_troops}
                num_territories = 42
                total_troops = 300      #arbitrary max number of troops a player can have

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

        def create_action_mask(game, query, q_values):
            match query:
                case "ClaimTerritory":
                    unclaimed_territories = game.state.get_territories_owned_by(None)
                    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
                    adjacent_territories = game.state.get_all_adjacent_territories(my_territories)
                    available = list(set(unclaimed_territories) & set(adjacent_territories))

                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    for territory in available:
                        mask[0, territory] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values

                case "PlaceInitialTroop":
                    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    for territory in my_territories:
                        mask[0, territory] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values

                case "RedeemCards":
                    cards_remaining = game.state.me.cards.copy()

                    comb_my_cards = combinations(cards_remaining.values(), 3)

                    # Filter valid sets
                    valid_sets = [combo for combo in comb_my_cards if game.state.get_card_set(combo) is not None]

                    valid_sets_ids = [[cards[i].card_id for i in combo] for combo in valid_sets]

                    # Create mask tensor
                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    for valid_set in valid_sets_ids:
                        mask[0, valid_set] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values


                case "DistributeTroops":

                    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
                    total_troops = game.state.me.troops_remaining

                    distributions = []
                    for i in range(my_territories):
                        for j in range(1, total_troops + 1):
                            distributions.append((i, j))

                    # Create mask tensor
                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    for distribution in distributions:
                        territory, troops = distribution
                        mask[0, territory, troops] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values

                case "Attack":
                    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
                    bordering_territories = game.state.get_all_adjacent_territories(my_territories)

                    attackable_territories = list(set(game.state.map.get_adjacent_to(bordering_territories)) & set(my_territories))
                    attack_from_territories = [territory for territory in my_territories if adjacent_territories & set(game.state.map.get_adjacent_to(territory))]

                    # Create mask tensor
                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    for from_territory in attack_from_territories:
                        for to_territory in attackable_territories:
                            for troops in range(1, 4):  # Attack with 1, 2, or 3 troops
                                mask[0, from_territory, to_territory, troops] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values

                case "TroopsAfterAttack":

                    record_attack = cast(RecordAttack, game.state.recording[query.record_attack_id])
                    move_attack = cast(MoveAttack, game.state.recording[record_attack.move_attack_id])

                    movable_troops = game.state.territories[move_attack.attacking_territory].troops - 1

                    # Create mask tensor
                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    mask[0, :movable_troops + 1] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values

                case "Defend":
                    num_troops_to_defend_with = 2
                    actions_tensor = torch.arange(num_troops_to_defend_with).int().unsqueeze(0)
                    return actions_tensor

                    move_attack = cast(MoveAttack, game.state.recording[query.move_attack_id])
                    defending_territory = move_attack.defending_territory
                    troops_on_defending_territory = defending_territory.troops

                    mask = torch.zeros(q_values.shape, dtype=torch.float32)

                    if troops_on_defending_territory < 2:
                        mask[0, 2] = 0
                        masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                        return masked_q_values
                    else:
                        return q_values

                case "Fortify":

                    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

                    unique_pairs = set()
                    actions = []

                    for territory in my_territories:
                        for adj_territory in game.state.map.get_adjacent_to(territory):
                            if (adj_territory, territory) not in unique_pairs:
                                unique_pairs.add((territory, adj_territory))
                            else:
                                continue
                            for troops in range(1, territory.troops - 1):
                                if (territory, adj_territory)
                                    actions.append((territory, adj_territory, troops))

                    mask = torch.zeros(q_values.shape, dtype=torch.float32)
                    for action in actions:
                        from_territory, to_territory, troops = action
                        mask[0, from_territory, to_territory, troops] = 1

                    masked_q_values = q_values * mask + (1 - mask) * float('-inf')
                    return masked_q_values


        """from_territory = np.random.choice(len(self.my_territories))
        to_territory = np.random.choice(len(self.adjacent_territories))
        num_troops = np.random.randint(1, self.max_troops + 1)
        return from_territory, to_territory, num_troops"""