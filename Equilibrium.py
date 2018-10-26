from micodymora.Chem import load_chems_dict
from micodymora.Reaction import Reaction
from micodymora.Constants import T0

import os
import numpy as np
from scipy.optimize import brentq

# tolerance for Brent's method when determining [H+]
H_tolerance = 1e-12
chems_dict_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "chems.csv")
chems_dict = load_chems_dict(chems_dict_path) # only needed by WaterEquilibrium

class Equilibrium:
    def __init__(self, chems, pK):
        self.chems = chems
        self.pK = [0] + pK
        # product of equilibrium constants from the first specie of the network
        # to each chem of the network
        self.Kprod = [10**-sum(self.pK[:i+1]) for i in range(len(self.pK))]
        # difference of charge with first species of the network
        self.charges = [chem.composition["charge"] for chem in self.chems]
        self.dgamma = [charge - self.charges[0] for charge in self.charges]

    def equilibrate(self, total, H):
        '''Computes the concentration of each chems according to proton
        concentration.
        * total: total concentration of all chems of the equilibrium
        * H: proton concentration
        '''
        numerators = [H**d * k for d, k in zip(self.dgamma, self.Kprod)]
        denominator = sum(numerators)
        return [total * numerator / denominator for numerator in numerators]

    def charge_balance(self, total, H):
        '''Returns the charge * concentration product of the species of the
        equilibrium'''
        concentrations = self.equilibrate(total, H)
        return sum(charge * conc for charge, conc in zip(self.charges, concentrations))

    def __str__(self):
        return "eq: " + " <-> ".join(str(chem) for chem in self.chems)

class WaterEquilibrium(Equilibrium):
    '''H2O <-> HO- + H+ equilibrium '''
    def __init__(self):
        self.chems = [chems_dict["HO-"]]
        self.pK = [0, 14]
        self.Kprod = [1, 1e-14]
        self.charges = [-1]
        self.dgamma = [0]

    def equilibrate(self, total, H):
        return [10**-self.pK[1] / H]

# take sumed concentrations vector
# returns flat concentrations vector
class SystemEquilibrator:
    def __init__(self, chems, equilibria, nesting):
        '''Instances of this class store all the equilibria to account for
        and apply them on aggregated concentration vectors.

        * chems: list of Chem instances in the expanded concentration vector
        * equilibria: list of Equilibrium instances to account for

            IMPORTANT: the list be be in the same order as the one given to
            the function determining the list of the chemical species involved
            in the simulation
        
        * nesting: a list of increasing integers indicating how species are
        aggregated in the aggregated concentration vector (see the Nesting
        module)
        '''
        # store the index of H+ and HO- in the aggregated vector
        H_index_in_expanded = next(i for i, chem in enumerate(chems) if chem.name == "H+")
        self.H_index = nesting.index(H_index_in_expanded)
        HO_index_in_expanded = next(i for i, chem in enumerate(chems) if chem.name == "HO-")
        self.HO_index = nesting.index(HO_index_in_expanded)

        # The `equilibria` attribute is organized as a list of Equilibrium
        # instances. One per chemical species in the aggregated concentration
        # vectors. If a chemical species is not involved in an equilibrium, it
        # is represented by a trivial equilibrium of one species.
        self.equilibria = list()
        eq_gen = (equilibrium for equilibrium in equilibria)
        # for each possible index in the aggregated vector:
        for i in range(len(nesting) - 1):
            # if it correspond to an equilibrium, assign it one of the
            # defined equilibria
            if nesting[i+1] - nesting[i] > 1:
                self.equilibria.append(next(eq_gen))
            # if it is HO-, assign it the special HO- equilibrium
            elif i == self.HO_index:
                self.equilibria.append(WaterEquilibrium())
            # otherwise create a trivial one-species equilibrium
            else:
                trivial_equilibrium = Equilibrium([chems[nesting[i]]], [0])
                self.equilibria.append(trivial_equilibrium)

    def charge_balance(self, H_guess, concentrations):
        '''Takes an aggregated concentration vector, computes the
        expanded equilibrated concentration vector according to the
        imposed H+ concentration, and return the charge balance of the
        expanded vector (as a float)'''
        concentrations[self.H_index] = H_guess
        return sum(eq.charge_balance(conc, H_guess) for conc, eq in zip(concentrations, self.equilibria))

    def equilibrate(self, concentrations):
        '''Takes an aggregated concentration vector, returns an expanded
        concentration vector with the concentration of all reagents, including
        H+, having been equilibrated'''
        def equilibrate_all(concentrations):
            '''Takes an aggregated concentration vector, returns an expanded
            concentration vector with the concentration of all reagents having
            been equilibrated according to the H+ concentration in the vector'''
            H_concentration = concentrations[self.H_index]
            equilibrated_concentrations = list()

            for concentration, equilibrium in zip(concentrations, self.equilibria):
                eq_result = equilibrium.equilibrate(concentration, H_concentration)
                equilibrated_concentrations.extend(eq_result)
            return equilibrated_concentrations
        # determine the value of [H+] for which the charge balance is null, 
        # while accounting for the equilibria
        H_root = brentq(self.charge_balance, 1e-14, 1, args=(concentrations,), xtol = H_tolerance)
        # set [H+] and determine the equilibrium concentrations
        concentrations[self.H_index] = H_root
        concentrations = equilibrate_all(concentrations)
        return concentrations

def load_equilibria_dict(chems_path, equilibria_path):
    '''Expected format of the equilibrium data file:
    on each row:
    - name of the equilibrium
    - list of the names of chemical species in their different form (space
      separated)
    - list of the pK (space separated)
    All those fields are separated by ":"
    acetate: Ac Ac(-): 4.78
    It is also possible to insert comments by beginning the line by "#"
    '''
    chems_dict = load_chems_dict(chems_path)
    equilibria_dict = dict()
    with open(equilibria_path, "r") as fh:
        for line in fh:
            if not line.startswith("#"):
                name, reagents, pKs_str = [field.lstrip().rstrip() for field in line.split(":")]
                chems = [chems_dict[reagent] for reagent in reagents.split(" ")]
                pKs = [float(pK) for pK in pKs_str.split(" ")]
                equilibria_dict[name] = Equilibrium(chems, pKs)
    return equilibria_dict

if __name__ == "__main__":
    equilibria_dict = load_equilibria_dict("data/chems.csv", "data/equilibria.dat")
    chems_names = ["H+", "HO-", "Ac", "Ac(-)", "Lac", "Lac(-)", "H3PO4", "H2PO4-", "HPO4-2", "PO4-3", "Na+", "CO2(aq)", "HCO3-", "CO3-2"]
    chems = [chems_dict[name] for name in chems_names]
    equilibria = [equilibria_dict[eq] for eq in ["acetate", "lactate", "phosphate", "carbonate"]]
    nesting = [0, 1, 2, 4, 6, 10, 11, 14]
    se = SystemEquilibrator(chems, equilibria, nesting)
    # we assume that phosphate is introduced as Na2HPO4 and carbonate as Na2CO3
    # and that they totally dissolve and never reprecipitate with Na+
    conc = [1e-7, 1e-7, 1e-3, 1e-3, 1e-3, 4e-3, 1e-3]
    eq_conc = se.equilibrate(conc)
    print("pH: {:.2f}".format(-np.log10(eq_conc[0])))
    for chem, conc in zip(chems, eq_conc):
        print("{}: {:2.2e}".format(chem.name, conc))
