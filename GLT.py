"""
# responsibility

This module contains classes designed to implement Gas-Liquid Transfers (GLT).

* GasLiquidTransfer instances represent specific GLTs (such as H2(g) <->
  H2(aq)) in general

* SimulationGasLiquidTransfer instances represent GLTs in a specific simulation
  (knowing the index of their gas and liquid phase in the simulation's
  concentration vector, and knowing their kla)

* SystemGasLiquidTransfers contains all the GLTs to account for in a simulation,
  it triggers the computation of all GLT dynamics by asking all its
  SimulationGasLiquidTransfer instances, and it serves as an interface with
  the Simulation instance.

# GLT rate expression

The transfer rate from gas to liquid is

> kla * (L - G * H)

with kla the volumetric mass transfer coefficient (h-1), L the concentration of
species in the liquid phase (M), G the concentration of species in the gas
phase (M) and H the Henry constant (Mliq.Mgas-1).

# Gas outflow rate

The GLT dynamics also include an expression for a valve in the culture vessel.
The expression of the outflow by the valve is

> alpha * (sum partial_pressure - headspace_pressure)

It occurs only if the sum of partial pressures in the vessel is superior to the
headspace pressure, so gas can only flow out, not in. The alpha parameter is
set to 0 by default, so the valve actually works only if explicitely decided by
the user in the configuration file.

# behaviour towards spurrious input

The computation of the transfer rates cannot give rise to null or negative
concentrations from positive concentrations, because the dynamics are of the
form C(t) = C0 exp(kt).

However, make sure that the input contains no null or negative values; it is
not this module's responsibility to ensure that and it won't produce consistent
results then.  Indeed, if the concentration of one of the two phases is null or
negative, the other phase will transfer into it to equilibrate it.  If both
phases are null or negative, GLT will produce artefactual oscillations between
the two phases (gas goes down, liquid goes up until it reaches zero or more,
then liquids transfers back to gas and goes negative again, and the cycle
repeats).
"""

from micodymora.Constants import T0, R
from micodymora.Chem import load_chems_dict

import numpy as np

class GasLiquidTransfer:
    def __init__(self, gas_chem, liquid_chem, H0cp, Hsol):
        '''
        * reaction: Reaction instance
        * H0cp: Henry constant (Mliq.Mgas-1) in standard conditions
        * Hsol: Enthalpy of solution over R (K)'''
        self.gas_chem = gas_chem
        self.liquid_chem = liquid_chem
        self.H0cp = H0cp
        self.Hsol = Hsol

class SimulationGasLiquidTransfer(GasLiquidTransfer):
    def __init__(self, gas_chem, liquid_chem, H0cp, Hsol, chems_list, kla, vliq, vgas):
        '''
        * reaction: Reaction instance. Is upgraded to a SimulationReaction
        instance during the instanciation.
        * kla: gas/liquid transfer rate constant (day-1)
        * chems_list: list of the name of the chemical species involved in
        the simulation, in the same order as in the concentration vector
        * vliq: volume of the liquid phase of the system
        * vgas: volume of the gas phase of the system
        '''
        # first initialise the instance as a GasLiquidTransfer
        super().__init__(gas_chem, liquid_chem, H0cp, Hsol)
        self.kla = kla
        # then get the index of the gas and liquid species as they
        # are needed by the rate function
        self.gas_index = chems_list.index(self.gas_chem.name)
        self.liquid_index = chems_list.index(self.liquid_chem.name)
        self.stoichiometry_vector = np.zeros(len(chems_list))
        self.stoichiometry_vector[self.gas_index] = 1 * vliq / vgas
        self.stoichiometry_vector[self.liquid_index] = -1
        
    def get_vector(self):
        return self.stoichiometry_vector

    def get_rate(self, C, T, tracker):
        """Compute the transfer rate from the gas phase to the liquid phase"""
        Cg = C[self.gas_index]
        Cl = C[self.liquid_index]
        H = self.H0cp * np.exp(self.Hsol * (1/T - 1/T0))
        return self.kla * (Cl - Cg * H * R * T)

    def get_gas_concentration(self, C):
        """return the concentration of the gas species (which is in mol.L-1)"""
        return C[self.gas_index]

    @classmethod
    def from_GasLiquidTransfer(cls, glt, chems_list, kla, vliq, vgas):
        '''Constructor from GasLiquidTransfer'''
        return cls(glt.gas_chem, glt.liquid_chem, glt.H0cp, glt.Hsol, chems_list, kla, vliq, vgas)

class SystemGasLiquidTransfers:
    def __init__(self, chems_list, transfers, vliq, vgas, headspace_pressure, alpha):
        '''
        * transfers: list of SimulationGasLiquidTransfer instances
        * vliq: volume of liquid phase (L)
        * vgas: volume of gas phase (L)
        * headspace_pressure: headspace space (Pa)
        * alpha: coefficient characterizing the outflow from the gas phase
          through a valve'''
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
        equilibration_rate = np.hstack(transfer.get_rate(C, T, tracker) for transfer in self.transfers)
        # gas outflow through a valve (only enabled if the alpha parameter is set to a non-zero value)
        # total pressure (in Pa) is computed by summing individual pressures,
        # which are computed from individual gas concentrations multiplied by RT
        # to convert them to J.L-1 then multiplied by 1e3 to be in J.m-3 (Pa)
        total_gas_pressure = sum(transfer.get_gas_concentration(C) * R * T * 1e3 for transfer in self.transfers)
        # NOTE: the water vapour pressure should be accounted for but is ignored for the moment
        gas_outflow_rate = np.array([self.alpha * (total_gas_pressure - self.headspace_pressure) * (total_gas_pressure > self.headspace_pressure)]) 
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
