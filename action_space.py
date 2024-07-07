import numpy as np
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

    def choose_random_action(self):
        match self.query:
            case "ClaimTerritory":
                chosen_territory_to_claim = np.random.choice(len(self.adjacent_territories))
                return chosen_territory_to_claim
            case "PlaceInitialTroop":
                chosen_territory_to_place_troop = np.random.choice(len(self.my_territories))
                troops_to_place = np.random.randint(1, self.total_troops + 1)
                return chosen_territory_to_place_troop, troops_to_place
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