from micodymora.Chem import load_chems_dict
from micodymora.Constants import Rkj, T0

import re
import numbers
import collections
import numpy as np

import pdb

class UndefinedQuantity:
    '''This class represents a quantity which has a physical meaning, but which
    cannot be represented by a number in a specific case because informations
    are lacking. For example, the enthalpy differential of a reaction.'''
    pass

class ImbalancedReactionException(Exception):
    pass

class Reaction:
    def __init__(self, reagents, name = None):
        '''reagents: {Chem: stoichiometry}'''
        self.reagents = reagents
        self.name = name
        self.dG0 = sum(chem.dGf0 * stoichiometry for chem, stoichiometry in self.reagents.items())
        try:
            self.dH0 = sum(chem.dHf0 * stoichiometry for chem, stoichiometry in self.reagents.items())
        except TypeError:
            self.dH0 = UndefinedQuantity

    def K(self, T):
        '''equilibrium constant of the reaction'''
        return np.exp(-self.dG0 / Rkj / T)

    def dG0p(self, T):
        '''Gibbs energy differential of the reaction (kJ.mol-1) assuming that
        every chemical species is at standard concentration, except for H+
        which is at 1e-7 M.'''
        try:
            proton_stoich = next(stoich for chem, stoich in self.reagents.items() if chem.name == "H+")
        except StopIteration:
            proton_stoich = 0
        return self.dG0 + Rkj * T * np.log(1e-7 ** proton_stoich) 

    def check_balance(self, tolerance=0.01):
        '''Checks the elemental balance of the reaction.
        Throws ImbalancedReactionException if reaction is imbalanced.
        Does not do anything remarkable if the reaction is balanced.
        Returns a dictionary containing the imabalance per element (0 if balanced)
        '''
        balance = collections.defaultdict(int)
        for reagent, stoichiometry in self.reagents.items():
            for element, amount in reagent.composition.items():
                balance[element] += stoichiometry * amount
        imbalances = [(element, imbalance) for element, imbalance in balance.items() if imbalance]
        if sum(imbalance for element, imbalance in imbalances) > tolerance:
            imbalance_summary = ", ".join("{}: {}".format(element, imbalance) for element, imbalance in imbalances)
            raise ImbalancedReactionException(imbalance_summary)
        return balance

    @classmethod
    def from_string(cls, chems_dict, reaction_string, name=None):
        '''Creates a Reaction instance from a string.
            The string must comply to the following requirements;
            - reagents are written as specified in the `chems_dict` used
                (see the "name" column in data/chems.csv for example)
            - substrates on the left hand side and products on the right hand side
            - both sides are separated by " --> "
            - reagents are separated by a " + " (spaces are mandatory)
            - stoichiometry of reagents without space (good: "2H2O"; bad: "2 H2O")
            It is possible to instanciate reactions with no product or no
            substrate.
            '''
        reaction_dict = dict()
        substrates_string, products_string = reaction_string.split("-->")
        substrates_string = substrates_string.lstrip().rstrip()
        products_string = products_string.lstrip().rstrip()
        chem_pat = re.compile("^([\d\.]+)?(\S+)")
        if substrates_string:
            individual_substrate_strings = substrates_string.split(" + ")
            for individual_substrate_string in individual_substrate_strings:
                stoichiometry, chem_name = chem_pat.findall(individual_substrate_string)[0]
                chem = chems_dict[chem_name] # raises an error if the chem in the reaction is not recorded in chems.csv
                stoichiometry = stoichiometry and -float(stoichiometry) or -1
                reaction_dict[chem] = stoichiometry
        if products_string:
            individual_product_strings = products_string.split(" + ")
            for individual_product_string in individual_product_strings:
                stoichiometry, chem_name = chem_pat.findall(individual_product_string)[0]
                chem = chems_dict[chem_name] # raises an error if the chem in the reaction is not recorded in chems.csv
                stoichiometry = stoichiometry and float(stoichiometry) or 1
                reaction_dict[chem] = stoichiometry
        return cls(reaction_dict, name=name)

    def normalize(self, chem_name):
        '''Normalize the stoichiometry of the reaction so that the 
        stoichiometric coefficient of a specific chem is 1 or -1
        '''
        # Get the Chem instance. If it fails, it means that the chem is absent
        # from the reaction
        chem = next(chem for chem in self.reagents.keys() if chem.name == chem_name)
        factor = abs(self.reagents[chem])
        self.reagents = {chem: coef / factor for chem, coef in self.reagents.items()}

    def new_SimBioReaction(self, chems_list, gas_species, parameters):
        '''Return a SimBioReaction instance based on the current Reaction
        instance'''
        return SimBioReaction(self.reagents, chems_list, gas_species, parameters, name=self.name)

    def __mul__(self, factor):
        '''Implement multiplication of reaction's stoichiometry by a numeric factor
        '''
        if isinstance(factor, numbers.Real):
            new_stoichiometry = {reagent: stoich * factor
                                 for reagent, stoich
                                 in self.reagents.items()}
            return Reaction(new_stoichiometry, self.name)
        else:
            raise ValueError("Attempt to multiply a {} instance by a {} object".format(type(self), type(factor)))

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __add__(self, operand):
        '''Implement addition for reactions'''
        if isinstance(operand, Reaction): # addition of two reactions
            new_stoichiometry = self.reagents.copy()
            for reagent in operand.reagents.keys():
                # compare chems names instead of Chem instances directly
                # because the same chem can correspond to different Chem
                # instances between two Reaction instances
                # eg: two reactions involving CO2 may have a "CO2" chem instance
                # as key in their `reagents` dict, but it may be two different
                # Chem instances bearing the same name
                try:
                    same_reagent = next(chem for chem in new_stoichiometry.keys() if chem.name == reagent.name)
                    new_stoichiometry[same_reagent] += operand.reagents[reagent]
                except StopIteration:
                    new_stoichiometry[reagent] = operand.reagents[reagent]
            new_name = "compound reaction"
        elif isinstance(operand, numbers.Real): # addition of real number to reaction
            new_stoichiometry = {reagent: stoich + operand
                                 for reagent, stoich
                                 in self.reagent.items()}
            new_name = self.name
        return Reaction(new_stoichiometry, new_name)

    def __getitem__(self, key):
        '''Get the stoichiometric coefficient of a reagent based on its name,
        returns 0 if the reagent is not involved
        '''
        try:
            return next(coef for chem, coef in self.reagents.items() if chem.name == key)
        except StopIteration:
            return 0

    def __str__(self):
        formatted_substrates = ["{}{}".format(stoichiometry != -1 and -stoichiometry or "", chem.name)
                                for chem, stoichiometry 
                                in self.reagents.items()
                                if stoichiometry < 0]
        substrates = " + ".join(formatted_substrates)
        formatted_products = ["{}{}".format(stoichiometry != 1 and stoichiometry or "", chem.name)
                                for chem, stoichiometry
                                in self.reagents.items()
                                if stoichiometry > 0]
        products = " + ".join(formatted_products)
        if self.name:
            return "{}: {} --> {}".format(self.name, substrates, products)
        else:
            return "{} --> {}".format(substrates, products)

class SimulationReaction(Reaction):
    '''Represents a chemical reaction occuring during a simulation.
    Instances of this class know the stoichiometric vector corresponding to
    their reaction, and also the rate function of the reaction.
    It is then able to compute its mass action ratios and Gibbs energy when
    given a concentration vector
    '''
    def __init__(self, reagents, chems_list, rate, name = None):
        super().__init__(reagents, name=name)
        self.chems_list = chems_list
        self.rate = rate
        self.stoichiometry_vector = np.zeros(len(chems_list))
        for reagent, stoichiometry in self.reagents.items():
            # if the reaction involves a reagent which is purposefully not
            # included in the simulation (eg: water), ignore it
            if reagent.name in chems_list:
                self.stoichiometry_vector[chems_list.index(reagent.name)] = stoichiometry

    def lnQ(self, C):
        '''Natural logarithm of the mass action ratio of the reaction.
        * C: concentrations vector (mol.L-1)'''
        # set null or negative concentrations to machine epsilon
        C = np.clip(C, np.finfo(float).eps, None)
        return sum(stoich * np.log(conc)
                    for stoich, conc
                    in zip(self.stoichiometry_vector, C))

    def dG(self, C, T):
        '''Compute Gibbs energy differential for non-standard conditions of
        temperature and concentrations, in kJ.mol-1.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)'''
        return self.dG0 + Rkj * T * self.lnQ(C)

    def get_vector(self, C, T):
        '''Returns the reaction's stoichiometric vector, containing values in
        molW.molD-1, where D is the substrate by which the reaction'
        stoichiometry is normalized, and W is whatever reagent of the reaction.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)'''
        return self.stoichiometry_vector

    def get_rate(self, C, T):
        '''Returns reaction's rate in molD.molX-1.day-1,
        where D is the substrate by which the reaction's stoichiometry is
        normalized, and X is the biomass catalyzing the reaction.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)'''
        return self.rate(C, T, self)

    @classmethod
    def from_reaction(cls, reaction, chems_list, rate):
        '''Constructor using a Reaction instance as basis'''
        return cls(reaction.reagents, chems_list, rate, name=reaction.name)

    @classmethod
    def from_string(cls, chems_dict, reaction_string, chems_list, rate, name=None):
        '''Constructor from string, overrides Reaction.from_string'''
        reaction = Reaction.from_string(chems_dict, reaction_string, name=name)
        return cls.from_reaction(reaction, chems_list, rate)

# The only difference between SimulationReaction and MetabolicReaction for the
# moment is that MetabolicReaction passes population informations to its rate
# instance
class MetabolicReaction(SimulationReaction):
    '''Represents a chemical reaction attached to a population in a simulation.
    Instances of this class know the stoichiometric vector corresponding to
    their reaction, the rate function of the reaction, and pass the informations
    from the population to which they belong to their rate function.
    It is then able to compute its mass action ratios and Gibbs energy when
    given a concentration vector
    '''
    def __init__(self, reagents, chems_list, rate, name = None):
        super().__init__(reagents, chems_list, rate, name=name)

    def get_rate(self, C, T, population):
        '''Returns reaction's rate in molD.molX-1.day-1,
        where D is the substrate by which the reaction's stoichiometry is
        normalized, and X is the biomass catalyzing the reaction.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)
        * population: population to which the reaction belongs'''
        return self.rate(C, T, self, population)

    @classmethod
    def from_reaction(cls, reaction, chems_list, rate):
        '''Constructor using a Reaction instance as basis'''
        return cls(reaction.reagents, chems_list, rate, name=reaction.name)

    @classmethod
    def from_simulation_reaction(cls):
        '''Constructor using a SimulationReaction instance as basis'''
        return cls(reaction.reagents, reaction.chems_list, reaction.rate, name=reaction.name)

    @classmethod
    def from_string(cls, chems_dict, reaction_string, chems_list, rate, name=None):
        '''Constructor from string, overrides Reaction.from_string'''
        reaction = Reaction.from_string(chems_dict, reaction_string, name=name)
        return cls.from_reaction(reaction, chems_list, rate)

# this class is meant to ultimately replace SimulationReaction and MetabolicReaction
class SimBioReaction(Reaction):
    def __init__(self, reagents, chems_list, gas_species, parameters, name=None):
        '''
        * reagents: {Chem: stoichiometry} dict
        * chems_list: list of the name of the chemical species (as strings)
        in the same order as in the concentrations vector
        * gas_species: boolean vector indicating whether a species is a gas or not
        '''
        super().__init__(reagents, name=name)
        self.chems_list = chems_list
        self.gas_species = gas_species
        self.stoichiometry_vector = np.zeros(len(self.chems_list))
        self.update_chems_list(chems_list) # the stoichiometry vector is set here
        self.parameters = parameters

    def update_stoichiometry_vector(self):
        '''The stoichiometry vector is stored by the instance as an attribute,
        since it is not expected to be modified after instanciation. However it
        may happen anyway. This function exists to update the stoichiometry
        vector after the reaction's stoichiometry has been modified'''
        self.stoichiometry_vector = np.zeros(len(self.chems_list))
        for reagent, stoichiometry in self.reagents.items():
            # if the reaction involves a reagent which is purposefully not
            # included in the simulation (eg: water), ignore it
            if reagent.name in self.chems_list:
                self.stoichiometry_vector[self.chems_list.index(reagent.name)] = stoichiometry

    def update_chems_list(self, new_chems_list):
        '''Change the chems list and recompute the stoichiometry vector accordingly'''
        self.chems_list = new_chems_list
        self.update_stoichiometry_vector()

    def set_stoichiometry_by_index(self, index, value, chem=None):
        '''Set to `value` the stoichiometric coefficient of the reagent whose
        index is `index` according to the reactions's chems list.
        If the reagent is not part of the reaction, a chem instance has to
        be created for it. The `chem` argument provides the Chem instance to use'''
        if self.chems_list[index] in [r.name for r in self.reagents]:
            reagent = next(reagent for reagent in self.reagents if reagent.name == self.chems_list[index])
            self.reagents[reagent] = value
        else:
            if chem:
                self.reagents[chem] = value
            else:
                raise ValueError("attempt to set the value of the reagent of index {} in reaction '{}' "
                                 "while it is not initially part of the reaction and no Chem instance "
                                 "has been provided for it".format(index, self.name))
        self.update_stoichiometry_vector()

    def lnQ(self, C):
        '''Natural logarithm of the mass action ratio of the reaction.
        * C: concentrations vector (mol.L-1)'''
        total_gas_mol = sum(self.gas_species * C)
        # convert gas species concentration to partial pressure
        # leave non-gas species as they are (mol.L-1)
        C_partial = C / ((1 ^ self.gas_species) + self.gas_species * total_gas_mol)
        # set null or negative concentrations to machine epsilon
        C_clipped = np.clip(C_partial, np.finfo(float).eps, None)
        return sum(stoich * np.log(conc)
                    for stoich, conc
                    in zip(self.stoichiometry_vector, C_clipped))

    def disequilibrium(self, C, T):
        '''Mass action ratio divided by equilibrium constant (Q/K)'''
        return np.exp(self.lnQ(C) - np.log(self.K(T)))

    def dG(self, C, T):
        '''Compute Gibbs energy differential for non-standard conditions of
        temperature and concentrations, in kJ.mol-1.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)'''
        return self.dG0 + Rkj * T * self.lnQ(C)

    def dGT(self, C, T):
        '''Compute standard state Gibbs energy differential for non-standard
        conditions of temperature and concentrations, in kJ.mol-1. Applies the
        correction for non-standard temperature which is the solution for the
        Gibbs-Helmoltz equation, as explained in Hanselmann 1991.  This formula
        requires the enthalpy of formation of every reagent to be known.
        It does not account for the deviation of the value of enthalpies of
        formation out of the standard conditions of temperature; correcting
        the enthalpies of formation would indeed require the heat capacity of
        each chemical species, which is not very easy to come by. Here we stick
        to Hanselman's assumption that enthalpy of formation does not change
        much in the temperature range of biological reactions.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)'''
        assert self.dH0 is not UndefinedQuantity
        dG0T = self.dG0 * T / T0 + self.dH0 * (T0 - T) / T0
        return dG0T + Rkj * T * self.lnQ(C)

    def get_vector(self, C, T):
        '''Returns the reaction's stoichiometric vector, containing values in
        molW.molD-1, where D is the substrate by which the reaction'
        stoichiometry is normalized, and W is whatever reagent of the reaction.
        * C: concentrations vector (mol.L-1)
        * T: temperature (K)'''
        return self.stoichiometry_vector

    def normalize(self, chem_name):
        super().normalize(chem_name)
        self.update_stoichiometry_vector()

    def __mul__(self, factor):
        new_reaction = super().__mul__(factor)
        return new_reaction.new_SimBioReaction(self.chems_list, self.gas_species, self.parameters)

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __add__(self, operand):
        # NOTE: when adding two SimBioReaction instances, the parameters of the
        # resulting reaction are the parameters of the leftmost reaction (I suppose)
        if isinstance(operand, SimBioReaction):
            assert self.chems_list == operand.chems_list
        new_reaction = super().__add__(operand)
        return new_reaction.new_SimBioReaction(self.chems_list, self.gas_species, self.parameters)

def load_reactions_dict(chems_path, reactions_path, dHf0_path=None):
    chems_dict = load_chems_dict(chems_path, dHf0_data_path=dHf0_path)
    with open(reactions_path, "r") as reactions_fh:
        reactions_dict = dict()
        for line in reactions_fh:
            reaction_dict = dict()
            reaction_name, reaction_string = line.rstrip().split(": ")
            reaction = Reaction.from_string(chems_dict, reaction_string, name=reaction_name)
            reaction.check_balance()
            reactions_dict[reaction_name] = reaction
    return reactions_dict
