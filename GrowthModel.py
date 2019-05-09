from micodymora.Constants import Rkj, T0, F

import abc
import numpy as np
import functools
import operator
import re

class GrowthModel(abc.ABC):
    '''Classes inheriting from GrowthModel represent a microbial population,
    and define its growth kinetics.

    Required attributes:
    * specific_chems: name of the specific chems for an instanciated population.
    * affected_by_dilution: boolean vector specifying whether the specific
    chems of the population are affected by the chemostat's dilution rate or not
    '''
    def __init__(self, population_name, chems_list, reactions, pathways, params):
        '''
        * population_name: name of the population (string)
        * chems_list: list of the name of the chems being part of the simulation
        (list of strings)
        (NOTE: at the moment of the instanciation of the GrowthModel, this list
        of chems does not contain the chems specific to the population, such
        as its biomass. They will be added later, through `register_chems`)
        * reactions: list of reactions catalyzed by the population (list of
        Reaction.SimBioReaction instances)
        (NOTE: at the moment of the instanciation of the GrowthModel, this
        list does not contain the reactions involving the population-specific
        chems, such as the anabolism, since those chems are not yet created.
        Again, they will be added through `register_chems`)
        * pathways: the pathways, as a dictionary, as extracted from the
          config file
        * params: dictionary of population- and model-specific parameters
        '''
        self.specific_chems = NotImplemented
        self.affected_by_dilution = NotImplemented
        super().__init__()

    @abc.abstractmethod
    def get_derivatives(self, expanded_y, T, tracker):
        '''Return a vector of derivatives for the concentration of all the
        chems in the system, depending on the model's logics.
        * expanded_y: concentrations vector of the system, in flat format
        (no nesting) (array of floats)
        * T: temperature of the system in Kelvin (float)
        * tracker: instance implementing Simulation.AbstractLogger
        '''
        pass
    
    @abc.abstractmethod
    def get_index_of_specific_chems_unaffected_by_dilution(self):
        '''Return the index of all the specific chems of the model not
        affected by the dilution rate of the system. The indexes
        are returned as a list of integers
        '''
        pass

    @abc.abstractmethod
    def register_chems(self, indexes, chems_list, specific_reactions):
        '''This message is supposed to be sent by the configuration parser,
        during the building of the simulation. At that point, the specific
        chems introduced by the growth model have been added to the
        concentration vector of the simulation, and the reactions involving
        those chems have been created by the parser, so the parser sends the
        specific chems' indexes and the specific reactions back to the GrowthModel
        instance through this message
        * indexes: dictionary mapping a chems' "role" name (eg: "biomass")
        to its index in the chems list of the simulation
        * chems_list: the complete chems list of the simulation, including the
        specific chems of all the species (list of strings)
        * specific_reactions: list of SimBioReaction instances containing
        the reactions involving specific chems
        The growth model instance is not supposed to be "mature" before
        receiving this message.'''
        pass

    @abc.abstractmethod
    def set_initial_concentrations(self, y0):
        '''Set the concentrations of the growth-model-specific variables
        in the initial concentrations vector'''
        pass

class ObjectivelessModel(GrowthModel):
    specific_chem_pat = re.compile("{([^{]+)}")
    # FIXME: UNSOLVED DESIGN QUESTIONS:
    # - the population needs the system's volume: how to transmit it?
    def __init__(self, population_name, chems_list, reactions, pathways, params):
        # model-specific parameters
        self.AXP_per_P = 1.06 # mol(ATP + ADP)/mol(proteins)
        self.initial_ATP_ADP_ratio = 7.6
        self.NADX_per_P = 0.12 # mol(NAD+ + NADH)/mol(proteins)
        self.initial_NAD_NADH_ratio = 7.8 # based on Andersen and von Meyenburg
        self.initial_metabolite_concentration = 1e-4 # Based on nothing
        self.FD = _rates_dict["MM"](chems_list, {}, params)
        self.FT = _rates_dict["energy threshold FT"](chems_list, {}, params)
        # population-specific parameters
        self.population_name = population_name
        self.chems_list = chems_list
        # 1. get the model's parameters
        self.X0 = params["X0"]
        # 2. get the model's specific chems
        # first: set the mandatory chems (intracellular protons,
        # ATP, NADH, ATP synthase, NADH deshydrogenase)
        self.specific_chems = {"pH": {"name": "{}_pH".format(self.population_name),
                                              "template": None,
                                              "initial concentration": 7},
                               "ATP": {"name": "{}_ATP".format(self.population_name),
                                               "template": "ATP"},
                               "ADP": {"name": "{}_ADP".format(self.population_name),
                                               "template": "ADP"},
                               "NADH": {"name": "{}_NADH".format(self.population_name),
                                                "template": "NADH"},
                               "NAD+": {"name": "{}_NAD+".format(self.population_name),
                                                "template": "NAD+"}}
        self.affected_by_dilution = [self.specific_chems["ATP"]["name"],
                                     self.specific_chems["NADH"]["name"]]
        self.protein_fractions = {"other": {"name": "{}_other_proteins".format(self.population_name),
                                            "membrane": False},
                                  "replication": {"name": "{}_replication_proteins".format(self.population_name),
                                                  "membrane": False},
                                  "NADH deshydrogenase": {"name": "{}_NADH_deshydrogenase".format(self.population_name),
                                                          "membrane": True},
                                  "ATP synthase": {"name": "{}_ATP_synthase".format(self.population_name),
                                                   "membrane": True}}
        # determine which reactions happen in the cell and which specific chems are involved
        specific_chems_names = set()
        for pathway_name, pathway_info in pathways.items():
            specific_chems_names.update(self.__class__.specific_chem_pat.findall(pathway_info["formula"]))
            self.protein_fractions[pathway_name] = {"name": "{}_{}_proteins".format(self.population_name, pathway_name.replace(" ", "_")),
                                                    "membrane": False}
        for specific_chem in specific_chems_names:
            if specific_chem not in self.specific_chems:
                if specific_chem in params["metabolites informations"]:
                    template = params["metabolites informations"].get(specific_chem).get("template")
                else:
                    template = None
                self.specific_chems[specific_chem] = {"name": "{}_{}".format(self.population_name, specific_chem.replace(" ", "_")),
                                                      "template": template}
            self.affected_by_dilution.append(self.specific_chems[specific_chem]["name"])
        for fraction_name, fraction_info in self.protein_fractions.items():
            self.specific_chems[fraction_name] = {"name": fraction_info["name"],
                                                  "template": None}
            self.affected_by_dilution.append(fraction_info["name"])

    def register_chems(self, indexes, chems_list, specific_reactions):
        self.chems_list = chems_list
        self.pathways = specific_reactions

        for pathway in self.pathways:
            # most metabolic reactions involve metabolites; the parameters of
            # the reactions refer to them generically. Those generic references
            # must be changed to specific references
            pathway.parameters["Km"] = {(chem in self.specific_chems and self.specific_chems[chem]["name"] or chem): km
                                        for chem, km
                                        in pathway.parameters["Km"].items()}
            self.FD.prepare(pathway)
            self.FT.prepare(pathway)
        self.indexes = indexes

    def get_derivatives(self, expanded_y, T, tracker):
        return NotImplemented

    def set_initial_concentrations(self, y0):
        # initially give the same concentration for every fraction
        nbof_fractions = len(self.protein_fractions)
        initial_fraction_concentration = self.X0 / nbof_fractions
        for fraction_info in self.protein_fractions.values():
            y0[self.chems_list.index(fraction_info["name"])] = initial_fraction_concentration
        # initially give the same concentration for every metabolite and proteins
        for specific_chem_info in self.specific_chems.values():
            y0[self.chems_list.index(specific_chem_info["name"])] = self.initial_metabolite_concentration
        # conserved moieties are given "physiological" concentrations
        y0[self.chems_list.index(self.specific_chems["ATP"]["name"])] = self.X0 * self.AXP_per_P * (self.initial_ATP_ADP_ratio / (1 + self.initial_ATP_ADP_ratio))
        y0[self.chems_list.index(self.specific_chems["ADP"]["name"])] = self.X0 * self.AXP_per_P * (1 / (1 + self.initial_ATP_ADP_ratio))
        y0[self.chems_list.index(self.specific_chems["NAD+"]["name"])] = self.X0 * self.NADX_per_P * (self.initial_NAD_NADH_ratio / (1 + self.initial_NAD_NADH_ratio))
        y0[self.chems_list.index(self.specific_chems["NADH"]["name"])] = self.X0 * self.NADX_per_P * (1 / (1 + self.initial_NAD_NADH_ratio))
        return y0

    def get_index_of_specific_chems_unaffected_by_dilution(self):
        return [self.chems_list.index(chem_info["name"])
                for chem_kind, chem_info
                in self.specific_chems.items()
                if chem_kind not in self.affected_by_dilution]


class Stahl(GrowthModel):
    """Implementation of a purposefully simple multi-pathway growth model.
    The rate of a pathway i is `rcat_i = [X] * FD * FT`
    For simplicity purpose, FD is multiplicative Monod kinetics and FT is Jin
    and Bethke's factor.
    
    The following parameters must be defined for the population;
    - X0: initial biomass concentration (molaa.L-1)
    - chems dict path: (optional) path for the chems dict to be used when
      interpreting the anabolic reaction
    - Ym: true yield
    - decay: biomass decay rate (h-1)
    - anabolism: name of the population's reaction which is the anabolic reaction
    - biomass: the specie to be considered as biomass in the anabolic equation
    - anabolism electron donor: the electron donor in the catabolism
      or '' if the catabolism's electron donor is not involved in anabolism
    - dGatp: the threshold energy for the catabolic pathways
    
    The following parameters must be defined for each pathway;
    - vmax: maximum catalytic rate
    - Km: affinity for substrate (dictionary)
    - electron donor: the name of the electron donor
    - m: a factor to multiply the threshold energy to adjust it by pathway
    """
    def __init__(self, population_name, chems_list, reactions, pathways, params):
        self.population_name = population_name
        self.chems_list = chems_list
        # take the specific parameters of the model
        self.X0 = params["X0"] # initial biomass concentration
        self.decay = params["decay"]
        self.biomass = params["biomass"]
        self.an_eD = params["anabolism electron donor"]
        self.anabolism = params["anabolism"]
        # all the reactions which are not the anabolic reaction are considered
        # to be catabolic pathways
        self.pathways = reactions
        # create the specific chems
        self.specific_chems = {"biomass": {"name": "{}_biomass".format(self.population_name),
                                           "template": self.biomass}}
        self.affected_by_dilution = ["biomass"]
        # prepare the rate formula for each pathways
        self.FD = _rates_dict["MM"](chems_list, {}, params)
        self.FT = _rates_dict["energy threshold FT"](chems_list, {}, params)
        for pathway in self.pathways:
            pathway.normalize(pathway.parameters["electron donor"])
            self.FD.prepare(pathway)
            self.FT.prepare(pathway)

    def register_chems(self, indexes, chems_list, specific_reactions):
        # update the chems list
        self.chems_list = chems_list
        self.X = indexes["biomass"]
        # store the anabolic reaction, apart from the catabolic pathways
        self.anabolism = next(reaction for reaction in specific_reactions if reaction.name == self.anabolism)
        # update the stoichiometry vector of the reactions
        for pathway in self.pathways:
            pathway.update_chems_list(self.chems_list)
        # the anabolic reaction can now be updated
        self.anabolism.update_chems_list(self.chems_list)
        # Noguera's Mdc factor is determined based on biomass' stoichiometric coefficient
        # If "electron donor" is set to null in the config file, it is assumed that it
        # is not involved in the anabolic reaction, so no normalization is done
        if self.an_eD: 
            self.anabolism.normalize(self.an_eD)
        self.Mdc = 1 / self.anabolism[self.specific_chems["biomass"]["name"]]

        # store the catabolism matrix now we know the length of the vectors
        self.reaction_matrix = np.vstack(pathway.stoichiometry_vector
                                         for pathway
                                         in self.pathways)

    def get_index_of_specific_chems_unaffected_by_dilution(self):
        return [self.chems_list.index(chem_info["name"])
                for chem_kind, chem_info
                in self.specific_chems.items()
                if chem_kind not in self.affected_by_dilution]

    def add_reaction(self, reaction):
        self.pathways.append(reaction)

    def set_initial_concentrations(self, y0):
        y0[self.X] = self.X0
        return y0

    def get_stoichiometry(self, pathway, y, T, tracker):
        eD = pathway.parameters["electron donor"]
        Ym = pathway.parameters["Ym"]
        Mdc = self.Mdc
        Rc = self.anabolism.stoichiometry_vector
        Re = pathway.stoichiometry_vector
        Rcd = pathway[eD]
        return Ym * Mdc * Rc + (1 + Rcd * Ym * Mdc) * Re

    def get_derivatives(self, y, T, tracker):
        X = y[self.X]
        rcat = np.hstack(X * self.FD.rate(pathway, y, T) * self.FT.rate(pathway, y, T)
                         for pathway in self.pathways)
        # each pathway is associated with a anabolism and catabolism stoichiometry
        R = [self.get_stoichiometry(pathway, y, T, tracker) for pathway in self.pathways]
        derivatives = np.sum(stoech * rate for stoech, rate in zip(R, rcat))
        derivatives[self.X] -= self.decay * X
        return derivatives

class SimpleGrowthModel(GrowthModel):
    """
    The following parameters must be defined for the population;
    - X0: initial biomass concentration (molaa.L-1)
    - chems dict path: (optional) path for the chems dict to be used when
      interpreting the anabolic reaction
    - energy barrier: total energy cost of biomass replication (dissipation + anabolism)
    - anabolism: name of the population's reaction which is the anabolic reaction
    - biomass: the specie to be considered as biomass in the anabolic equation
    (kJ.molX-1, negative)
    - decay: negative exponential biomass decay coefficient (positive number, hour-1)
      decay contribution to dX/dt: -decay * X
    
    The following parameters must be defined for each pathway;
    - vmax: maximum catalytic rate
    - Km: affinity for substrate (dictionary)
    - norm: the chemical species by which the yield on the pathway is normalized
    (The "s" of the Yx/s. Usually the electron donor)
    """
    def __init__(self, population_name, chems_list, reactions, reactions_from_config, params):
        self.population_name = population_name
        self.chems_list = chems_list
        # take the specific parameters of the model
        self.X0 = params["X0"] # initial biomass concentration
        self.decay = params["decay"] # negative exponential decay coefficient
        self.biomass = params["biomass"]
        self.anabolism = params["anabolism"]
        # all the reactions which are not the anabolic reaction are considered
        # to be catabolic pathways
        self.pathways = reactions
        # create the specific chems
        self.specific_chems = {"biomass": {"name": "{}_biomass".format(self.population_name),
                                           "template": self.biomass}}
        self.affected_by_dilution = ["biomass"]
        # prepare the rate formula for each pathways
        self.FD = _rates_dict["MM"](chems_list, {}, params)
        self.FT = _rates_dict["energy threshold FT"](chems_list, {}, params)
        self.energy_barriers = np.zeros(len(self.pathways))
        for index, pathway in enumerate(self.pathways):
            self.FD.prepare(pathway)
            self.FT.prepare(pathway)
            self.energy_barriers[index] = pathway.parameters["energy barrier"]

    def register_chems(self, indexes, chems_list, specific_reactions):
        # update the chems list
        self.chems_list = chems_list
        self.X = indexes["biomass"]
        # store the anabolic reaction, apart from the catabolic pathways
        self.anabolism = next(reaction for reaction in specific_reactions if reaction.name == self.anabolism)
        # save a vector of the anabolic reaction stoichiometry that have the right shape for numpy computations
        # (the anabolic stoichiometry vector duplicated for each pathway and stacked into rows)
        self.anabolism_stoichiometry_vector = np.row_stack(self.anabolism.stoichiometry_vector for i in range(len(self.pathways)))
        # update the stoichiometry vector of the reactions
        for pathway in self.pathways:
            pathway.update_chems_list(self.chems_list)
        # the anabolic reaction can now be updated
        self.anabolism.update_chems_list(self.chems_list)
        self.anabolism.normalize(self.specific_chems["biomass"]["name"])

        # store the catabolism matrix now we know the length of the vectors
        self.reaction_matrix = np.row_stack(pathway.stoichiometry_vector
                                            for pathway
                                            in self.pathways)

    def get_index_of_specific_chems_unaffected_by_dilution(self):
        return [self.chems_list.index(chem_info["name"])
                for chem_kind, chem_info
                in self.specific_chems.items()
                if chem_kind not in self.affected_by_dilution]

    def add_reaction(self, reaction):
        self.pathways.append(reaction)

    def set_initial_concentrations(self, y0):
        y0[self.X] = self.X0
        return y0

    def get_derivatives(self, y, T, tracker):
        X = y[self.X]
        dG = np.column_stack(pathway.dG(y, T) for pathway in self.pathways)
        rcat = np.column_stack(self.FD.rate(pathway, y, T) * self.FT.rate(pathway, y, T)
                               for pathway in self.pathways)
        JG = rcat * dG
        # rate of energy intake from the environment
        ran = JG / np.clip(self.energy_barriers - self.anabolism.dG(y, T), a_max=-1, a_min=None) - self.decay * X / len(self.pathways)
        # each pathway is associated with a anabolism and catabolism stoichiometry
        metabolism = np.dot(rcat, self.reaction_matrix) + np.dot(ran, self.anabolism_stoichiometry_vector)
        derivatives = X * metabolism
        return derivatives

class RateFunction(abc.ABC):
    '''Classes inheriting from this abstraction represent functions computing
    part or all of the rate of a chemical reaction.
    
    Interface:
    Rate functions are purposed to work on SimBioReaction instances.
    What they require depends from the concrete implementation of the
    rate function, but they usually require the object to have a
    `parameters` field, containing parameters useful to the rate function.
    The reactions may also be asked to compute their dG.

    Before the beginning of a simulation, the rate function inspect the
    reactions to make sure they have sufficient data in their `parameter`
    field, and eventually to provide them with new ones. This process
    is done by the `prepare` function

    During the simulation, the rate function is given reaction instances
    and compute their numerical result from the parameters found inside
    the `parameter` field of the objects.
    '''
    def __init__(self, chems_list, parameters, growth_model_parameters):
        super().__init__()

    @abc.abstractmethod
    def prepare(self, reaction):
        '''Make sure that the reaction instance has what is required to be
        used by the rate function'''
        pass

    @abc.abstractmethod
    def rate(self, reaction, y, T):
        '''Compute the numerical value of the rate function, based on the
        state of the system (y, T) and the parameters of the reaction'''
        pass

class MM_kinetics(RateFunction):
    '''Rate of a chemical reaction according to irreversible, multiplicative
    Michaelis-Menten kinetics.

    Expected pathway parameters:
    * vmax: maximum reaction rate (hour-1)
    * Km: half-saturation concentration for the limiting substrate (M),
    set as a dictionary mapping the name of a limiting substrate to its
    Km value.
    '''
    def __init__(self, chems_list, parameters, growth_model_parameters):
        self.chems_list = chems_list

    def prepare(self, pathway):
        assert "vmax" in pathway.parameters
        assert "Km" in pathway.parameters
        # add a new parameter to the pathway, which is a dictionary mapping
        # limiting chems indexes to their respective Km
        pathway.parameters["MMKm"] = {self.chems_list.index(limiting): km
                                      for limiting, km
                                      in pathway.parameters["Km"].items()}

    def rate(self, pathway, C, T):
        vmax = pathway.parameters["vmax"]
        km_couples = pathway.parameters["MMKm"].items()
        return vmax * functools.reduce(operator.mul, (C[i] / (C[i] + k) for i, k in km_couples))

# WARNING: untested
class HohCordRuwischRate(RateFunction):
    '''
    Mono-substrate MM-based reaction rate accounting for thermodynamic limitation

    Expected parameters:
    * vmax: maximum reaction rate (hour-1)
    * Km: half-saturation concentration for the limiting substrate (M)
    * kr: reverse factor
    * limiting: name of the limiting substrate

    Expected pathway methods:
    * disequilibrium(C, T): the ratio of the reaction's mass action ratio over
    its equilibrium constant (Q/K)
    '''
    def __init__(self, chems_list, parameters, growth_model_parameters):
        self.chems_list = chems_list

    def prepare(self, pathway):
        assert "vmax" in pathway.parameters
        assert "kr" in pathway.parameters
        assert "Km" in pathway.parameters
        assert "limiting" in pathway.parameters
        pathway.parameters["limiting index"] = self.chems_list.index(pathway.parameters["limiting"])

    def rate(self, pathway, C, T):
        disequilibrium = pathway.disequilibrium(C, T)
        S = C[pathway.parameters["limiting index"]]
        Km = pathway.parameters["Km"]
        kr = pathway.parameters["kr"]
        vmax = pathway.parameters["vmax"]
        return vmax * S * (1 - disequilibrium) / (Km + S * (1 + kr * disequilibrium))

class JinBethkeFT(RateFunction):
    '''
    [0-1] thermodynamic rate limitation factor based on Boudart's model
    (Boudart, 1976).

    Expected growth model parameters:
    * dGatp: Gibbs energy differential to consider for the hydrolysis of ATP

    Expected pathway parameters:
    * m: number of ATP molecules produced by the pathway (molATP.turnover-1)
    * chi: average stoichiometric coefficient of the reaction, accounting for
    the fact that the reaction actually consists in multiple steps running at
    different speeds. Assumed to be 1 if not set.

    Expected pathway methods:
    * dG(C, T): the Gibbs energy differential of the pathway, ajusting it for
    non-standard temperature and concentrations
    '''
    def __init__(self, chems_list, parameters, growth_model_parameters):
        '''
        * dGatp: Gibbs energy differential to consider for the hydrolysis of ATP
        '''
        self.chems_list = chems_list
        self.dGatp = growth_model_parameters["dGatp"]

    def prepare(self, pathway):
        assert "m" in pathway.parameters
        assert "dG" in dir(pathway)
        if "chi" not in pathway.parameters:
            pathway.parameters["chi"] = 1

    def rate(self, pathway, C, T):
        dissipation = pathway.dG(C, T) - pathway.parameters["m"] * self.dGatp
        if dissipation > 0:
            dissipation = 0
        return 1 - np.exp(dissipation / pathway.parameters["chi"] / T / Rkj)

class energy_threshold_FT(RateFunction):
    '''
    FT factor as implemented by Noguera, Brusseau, Rittman and Stahl
    (doi: 10.1002/(SICI)1097-0290(19980920)59:6<732::AID-BIT10>3.0.CO;2-7)

    FT = (1 - exp((dGr - dGmin) / RT))

    Expected pathway parameters:
    * dGmin: minimum energy threshold for a pathway to run

    Expected pathway methods:
    * dG(C, T): the Gibbs energy differential of the pathway, ajusting it for
    non-standard temperature and concentrations
    '''
    def __init__(self, chems_list, parameters, growth_model_parameters):
        self.chems_list = chems_list

    def prepare(self, pathway):
        assert "dGmin" in pathway.parameters

    def rate(self, pathway, C, T):
        dissipation = pathway.dG(C, T) - pathway.parameters["dGmin"]
        if dissipation > 0:
            dissipation = 0
        return 1 - np.exp(dissipation / T / Rkj)

# WARNING: not validated
class LaRowe2012FT(RateFunction):
    '''
    Expected pathway parameters:
    * dPsi: electrical potential of the cell (mV)
    * dgamma: number of transfered electrons in the catabolic reaction

    Expected pathway methods:
    * dG(C, T): the Gibbs energy differential of the pathway, ajusting it for
    non-standard temperature and concentrations
    '''
    def __init__(self, chems_list, parameters, growth_model_parameters):
        self.chems_list = chems_list

    def prepare(self, pathway):
        assert "dPsi" in pathway.parameters
        assert "dgamma" in pathway.parameters

    def rate(self, pathway, C, T):
        dGr = pathway.dG(C, T) / pathway.parameters["dgamma"]
        if dGr < 0:
            # 1e-6 factor to convert mV to V then J to kJ
            FT = 1 / (np.exp((dGr + F * pathway.parameters["dPsi"] * 1e-6) / Rkj / T) + 1)
        else:
            FT = 0
        return FT


_rates_dict = {"MM": MM_kinetics,
               "HohCordRuwisch": HohCordRuwischRate,
               "energy threshold FT": energy_threshold_FT,
               "JinBethkeFT": JinBethkeFT,
               "LaRowe2012FT": LaRowe2012FT}

growth_models_dict = {"Objectiveless": ObjectivelessModel,
                      "SimpleGrowthModel": SimpleGrowthModel,
                      "Stahl": Stahl,
                      }
