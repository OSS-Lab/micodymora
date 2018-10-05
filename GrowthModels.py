from abc import ABC

class GrowthModel(ABC):
    def __init__(self, reactions, params):
        pass

    def get_derivatives(self, expanded_y, T):
        pass

    def chems_to_register():
        pass

class ThermoAllocationModel(GrowthModel):
    def __init__(self, reactions, params):
        self.pathways = pathways
        self.params = params
