'''
All rate function classes must comply to the same interface:
- having their actual rate computation function taking the concentration vector
  as first argument and the system's temperature as second argument
- having their rate computation function overloaded on the class' __call__
- being initialized with the following arguments (in this order):
- - the list of the name of the chemical species in the expanded concentration
  vector, in the same order
- - the MetabolicReaction instance they belong to.
- - the rate function's parameters dictionary.
'''

# design note: it is expected that the `reaction` argument is not known at the
# moment of the instanciation of a rate instance, because the rate instance
# itself is a component of a MetabolicReaction instance. It will then be
# initialized as None during the "normal" instanciation of Rate. This attribute
# is later automatically set during MetabolicReaction's instanciation.
# It is kept as an argument in rate classes initialization because:
# - it makes all attributes of the rate class visible in __init__
# - it allows special instanciations (like instances copies etc)

from operator import mul
from functools import reduce
from Constants import T0
import math

class MM_rate:
    '''Rate of a chemical reaction according to irreversible, multiplicative
    Michaelis-Menten kinetics.
    '''
    def __init__(self, chems_list, reaction, params):
        '''
        Expected parameters:
        * vmax: maximum reaction rate (day-1)
        * Ks: {chem_name: Ks} dictionary
        '''
        self.reaction = reaction
        self.vmax = params["vmax"]
        # map the chem index instead of the chem name
        self.Ks = {chems_list.index(chem): float(ks) for chem, ks in params["Ks"].items()}
        self.description_str = "{} instance: {} * {}".format(self.__class__.__name__, self.vmax, " * ".join("{0} / ({0} + {1})".format(S, Ks) for S, Ks in params["Ks"].items()))

    def __call__(self, C, T):
        return self.vmax * reduce(mul, (C[i] / (C[i] + k) for i, k in self.Ks.items()))

    def __str__(self):
        return self.description_str

class HohCordRuwisch_rate:
    '''Single substrate reaction rate based on MM, accounting for
    thermodynamics. See Hoh and Cord-Ruwisch 1996.'''
    def __init__(self, chems_list, reaction, params):
        '''
        Expected parameters:
        * vmax: maximum reaction rate (day-1)
        * Ks: half-saturation concentration for the limiting substrate (M)
        * kr: reverse factor
        * limiting: name of the limiting substrate
        '''
        self.reaction = reaction
        self.limiting = chems_list.index(params["limiting"])
        self.vmax = params["vmax"]
        self.Ks = params["Ks"]
        self.kr = params["kr"]

    def __call__(self, C, T):
        diseq = math.exp(self.reaction.lnQ(C) - math.log(self.reaction.K(T)))
        S = C[self.limiting]
        return self.vmax * S * (1 - diseq) / (self.Ks + S * (1 + self.kr * diseq))

class empirical_anabolism_rate:
    '''Empirical population growth rate.
    The population-scale rate of each catabolism is in molD.t-1 (where D is the
    species by which the reaction's stoichiometry is normalized), and each
    catabolism is supposed to induce the production of biomass, considering
    a yield Y in molX.molD-1 specific to each catabolism. The overall growth
    rate of a population catalyzing multiple catabolisms is then the sum
    of all the anabolic rates associated to all individual catabolisms.
    '''
    def __init__(self, chems_list, reaction, params):
        '''
        Expected parameters:
        * Y: array of the yields of the catabolic reactions (molX.molD-1)
        (where D is the reagent by which the catabolism's stoichiometry is
        normalized)
        * kd: array of the decay rates associated to each catabolism 
        '''
        self.reaction = reaction
        self.Y = params["Y"]
        self.kd = params["kd"]
    
    def __call__(self, C, T):
        # design note: the law of demeter is broken
        catabolic_rates = self.reaction.population.get_catabolism_rates(C, T)
        return sum(self.Y * catabolic_rates - self.kd)

class gas_transfer_rate:
    '''Transfer rate of a chemical species between a gas phase and a liquid
    phase.'''
    def __init__(self, chems_list, reaction, params):
        '''Expected parameters:
        * H0cp: Henry constant (M.atm-1) in standard conditions
        * Hsol: Enthalpy of solution over R (K)
        * kla: gas/liquid transfer rate constant (day-1)'''
        self.reaction = reaction
        self.kla = params["kla"]
        self.H0cp = params["H0cp"]
        self.Hsol = params["Hsol"]
        self.gas_index = params["gas index"]
        self.liquid_index = params["liquid index"]

    def __call__(self, C, T):
        Cg = C[self.gas_index]
        Cl = C[self.liquid_index]
        H = self.H0cp * math.exp(1/T - 1/T0)
        return self.kla * (Cl - Cg * H)

rates_dict = {"MM": MM_rate,
              "Hoh96": HohCordRuwisch_rate,
              "empirical anabolism": empirical_anabolism_rate}
