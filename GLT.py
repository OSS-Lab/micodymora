from micodymora.Constants import T0
from micodymora.Chem import load_chems_dict

import numpy as np

class GasLiquidTransfer:
    def __init__(self, gas_chem, liquid_chem, H0cp, Hsol):
        '''
        * reaction: Reaction instance
        * H0cp: Henry constant (M.atm-1) in standard conditions
        * Hsol: Enthalpy of solution over R (K)'''
        self.gas_chem = gas_chem
        self.liquid_chem = liquid_chem
        self.H0cp = H0cp
        self.Hsol = Hsol

class SimulationGasLiquidTransfer(GasLiquidTransfer):
    def __init__(self, gas_chem, liquid_chem, H0cp, Hsol, chems_list, kla):
        '''
        * reaction: Reaction instance. Is upgraded to a SimulationReaction
        instance during the instanciation.
        * kla: gas/liquid transfer rate constant (day-1)
        * chems_list: list of the name of the chemical species involved in
        the simulation, in the same order as in the concentration vector
        '''
        # first initialise the instance as a GasLiquidTransfer
        super().__init__(gas_chem, liquid_chem, H0cp, Hsol)
        self.kla = kla
        # then get the index of the gas and liquid species as they
        # are needed by the rate function
        self.gas_index = chems_list.index(self.gas_chem.name)
        self.liquid_index = chems_list.index(self.liquid_chem.name)
        self.stoichiometry_vector = np.zeros(len(chems_list))
        self.stoichiometry_vector[self.gas_index] = 1
        self.stoichiometry_vector[self.liquid_index] = -1
        
    def get_vector(self):
        return self.stoichiometry_vector

    def get_rate(self, C, T, tracker):
        Cg = C[self.gas_index]
        Cl = C[self.liquid_index]
        H = self.H0cp * np.exp(self.Hsol * (1/T - 1/T0))
        return self.kla * (Cl - Cg * H)

    def get_partial_pressure(self, C):
        return C[self.gas_index]

    @classmethod
    def from_GasLiquidTransfer(cls, glt, chems_list, kla):
        '''Constructor from GasLiquidTransfer'''
        return cls(glt.gas_chem, glt.liquid_chem, glt.H0cp, glt.Hsol, chems_list, kla)

class SystemGasLiquidTransfers:
    def __init__(self, chems_list, transfers, vliq, vgas, headspace_pressure, alpha):
        '''
        * transfers: list of SimulationGasLiquidTransfer instances
        * vliq: volume of liquid phase (L)
        * vgas: volume of gas phase (L)
        * headspace_pressure: headspace space (atm)
        * alpha: coefficient related to gas flow'''
        self.transfers = transfers
        self.vliq = vliq
        self.vgas = vgas
        self.headspace_pressure = headspace_pressure
        self.alpha = alpha
        # also create and additional vector to account for
        # the process of gas flow out of the reactor
        self.gas_outflow_vector = np.zeros(len(chems_list))
        for transfer in transfers:
            self.gas_outflow_vector[transfer.gas_index] = -1

    def get_matrix(self):
        gl_transfers = np.vstack(transfer.get_vector() for transfer in self.transfers) 
        return np.vstack([gl_transfers, self.gas_outflow_vector])

    def get_rates(self, C, T, tracker):
        equilibration_rate = np.hstack(transfer.get_rate(C, T, tracker) for transfer in self.transfers) * self.vliq / self.vgas
        # should also account for water vapour pressure. Ignored for the moment
        total_gas_pressure = sum(transfer.get_partial_pressure(C) for transfer in self.transfers)
        gas_outflow_rate = np.array([self.alpha * (total_gas_pressure - self.headspace_pressure)]) 
        return np.hstack([equilibration_rate, gas_outflow_rate])

def load_glt_dict(chems_data_path, glt_data_path):
    chems_dict = load_chems_dict(chems_data_path)
    with open(glt_data_path, "r") as stream:
        glt_dict = dict()
        next(stream) # skip header
        for line in stream:
            if not line.startswith("#"):
                name, liquid, gas, H0cp, Hsol = line.rstrip().split(",")
                glt = GasLiquidTransfer(chems_dict[gas], chems_dict[liquid], float(H0cp), float(Hsol))
                glt_dict[name] = glt
    return glt_dict
