from micodymora.Constants import Rkj, T0, F

import abc
import numpy as np
import functools
import operator

class GrowthModel(abc.ABC):
    '''Classes inheriting from GrowthModel represent a microbial population,
    and define its growth kinetics.

    Required attributes:
    * specific_chems: name of the specific chems for an instanciated population.
    * affected_by_dilution: boolean vector specifying whether the specific
    chems of the population are affected by the chemostat's dilution rate or not
    '''
    def __init__(self, population_name, chems_list, reactions, params):
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

class ThermoAllocationModel(GrowthModel):
    # whatever the FT and FD functions used, the pathways should have an ATP
    # parameter defining the stoichiometry of ATP in the pathway
    def __init__(self, population_name, chems_list, reactions, params):
        self.population_name = population_name
        self.chems_list = chems_list
        self.pathways = reactions
        # stoichiometric coefficient for ATP in each pathway
        self.nu_atp_cat = np.array([pathway.parameters["m"] for pathway in self.pathways])
        # fixed fraction of housekeeping and structure proteins (molprotein.molbiomass-1)
        self.phi_x = params["phi x"]
        # fraction of enzymes involved in amino-acid biosynthesis (molprotein.molbiomass-1)
        self.phi_an = params["phi an"]
        # translation speed (molaminoacid.molribosome-1.hour-1)
        self.vt = params["vt"]
        # number of ATP consummed per aminoacid during the translation
        self.nu_atp_tr = params["nu atp tr"]
        # number of ATP consummed per aminoacid during their biosynthesis
        self.nu_atp_an = params["nu atp an"]
        # intracellular concentration ATP + ADP
        self.iap = params["intracellular adenosine phosphate"]
        # ATP:ADP ratio
        self.iar = params["intracellular atp to adp ratio"]
        # molar fraction of aminoacid in total (dry) biomass (molaminoacid.molbiomass-1)
        self.xaa = params["xaa"]
        # proteome reallocation inertia
        self.k = params["k"]
        # Gibbs energy differential of ATP hydrolysis (one ~P)
        self.dGatp = params["dGatp"]
        # affinity coefficient of protein translation for ATP (molATP.L-1)
        self.Katp = params["Katp"]
        # initial proteome concentration
        self.X0 = params["X0"]
        # rate functions
        self.FD = _rates_dict[params["fd"]["function"]](chems_list, params["fd"], params)
        self.FT = _rates_dict[params["ft"]["function"]](chems_list, params["fd"], params)

        for pathway in self.pathways:
            self.FD.prepare(pathway)
            self.FT.prepare(pathway)

        # specific_chems is an attribute made for the configuration parser,
        # to tell it what chems to create for the specific needs of the
        # population. 
        self.specific_chems = {"biomass": {"name": "{}_total_biomass".format(self.population_name),
                                           "template": None},
                               "phi r": {"name": "{}_phi_r".format(self.population_name),
                                         "template": None}}
        for pathway in self.pathways:
            self.specific_chems["phi cat {}".format(pathway.name)] = {"name": "{}_phi_cat_{}".format(self.population_name, pathway.name),
                                                                      "template": None}
        # only total proteome and atp should be subjected to dilution
        self.affected_by_dilution = ["biomass"]

    def get_index_of_specific_chems_unaffected_by_dilution(self):
        return [self.chems_list.index(chem_info["name"])
                for chem_kind, chem_info
                in self.specific_chems.items()
                if chem_kind not in self.affected_by_dilution]

    def register_chems(self, indexes, chems_list, specific_reactions):
        # The type of chem to which the indexes correspond is set by the order
        # of the chems in self.specific_chems
        self.X = indexes["biomass"]
        self.phi_r = indexes["phi r"]
        # following chems are for pathway-specific enzyme fractions, in the same order
        # as in self.pathway
        self.phi_cat = {chem_kind.replace("phi cat ", ""): index
                        for chem_kind, index
                        in indexes.items() if chem_kind.startswith("phi cat")}
        # update the stoichiometry vector of the reactions
        for pathway in self.pathways:
            pathway.update_chems_list(self.chems_list)
        # now the reaction matrix of the pathways can be stored
        self.reaction_matrix = np.vstack(pathway.stoichiometry_vector
                                         for pathway
                                         in self.pathways)

    def set_initial_concentrations(self, y0):
        y0[self.X] = self.X0
        y0[self.phi_r] = self.phi_x / 2
        for phi_cat in self.phi_cat.values():
            y0[phi_cat] = (1 - self.phi_x * 3 / 2) / len(self.phi_cat)
        return y0

    def scaled_phi(self, y):
        '''return a dictionary of the value of the different fractions (phi) of
        the proteome, scaled to 1'''
        # total phi is without phi x
        total_phi = y[self.phi_r] + sum(y[phi_cat] for phi_cat in self.phi_cat.values())
        print(self.phi_cat.items())
        cat = {name: y[phi_cat] / total_phi * (1 - self.phi_x - self.phi_an)
               for name, phi_cat 
               in self.phi_cat.items()}
        other = {"x": self.phi_x, "r": max(0, y[self.phi_r]) / total_phi * (1 - self.phi_x - self.phi_an)}
        scaled_phi_dict = other
        scaled_phi_dict.update(cat)
        return scaled_phi_dict

    def proteome_derivatives(self, y, T, cat_atp_flows, katp, tracker):
        '''Proteome reallocation strategy
        * y: system's concentrations vector
        * T: system's temperature
        * cat_atp_flows: atp flow rates (molatp.hour-1) for each catabolic pathway
        * katp: forward/backward flow ratio of ADP + Pi = ATP
        '''
        derivatives_vector = np.zeros(len(self.chems_list))
        # phi cat
        # if there is an atp flow, allocate the enzymes so as to maximize this atp flow
        # if there is no atp flow at all, enzymes fractions do not change, since the population
        # has no energy at all it cannot perform any reallocation
        total_atp_flow = sum(cat_atp_flows)
        if total_atp_flow:
            ideal_phi_cats = cat_atp_flows / total_atp_flow
            #tracker.update_log("\n".join("{}: {:.2e}".format(pathway.name, phi_star) for pathway, phi_star in zip(self.pathways, ideal_phi_cats)))
            for population, ideal_phi_cat in zip(self.pathways, ideal_phi_cats):
                phi_cat = y[self.phi_cat[population.name]]
                derivatives_vector[self.phi_cat[population.name]] = self.k * (ideal_phi_cat - phi_cat)

        # phi r
        derivatives_vector[self.phi_r] = self.k * y[self.phi_r] * (katp - self.iar)

        return derivatives_vector

    def rcat(self, y, T, tracker):
        scaled_phis = self.scaled_phi(y)
        rcat_list = list()
        X = y[self.X]
        for pathway in self.pathways:
            phi_cat = scaled_phis[pathway.name]
            enzymes_per_step = X * phi_cat / pathway.parameters["p"]
            FD = self.FD.rate(pathway, y, T)
            FT = self.FT.rate(pathway, y, T)
            rcat_list.append(enzymes_per_step * FD * FT)
        return np.hstack(rcat_list)

    def maintenance(self, T):
        '''Maintenance as implemented by Tijhuis et al., 1993 (kJ.C-molX-1.hour-1)'''
        return -4.5 * np.exp(69e3 / Rkj * (1 / T - 1 / T0))

    def get_derivatives(self, y, T, tracker):
        scaled_phis = self.scaled_phi(y)
        # get the vector of derivatives, without anabolism
        rcat = self.rcat(y, T, tracker)
        X = y[self.X]
        derivatives = np.matmul(rcat, self.reaction_matrix)
        # compute the ATP concentration
        cat_atp_flows = rcat * self.nu_atp_cat
        forward_atp_rate = sum(cat_atp_flows)
        backward_atp_rate = X * (self.maintenance(T) / self.xaa / self.dGatp
                                 + scaled_phis["r"] * self.vt * (self.nu_atp_tr + self.nu_atp_an))
        katp = forward_atp_rate / backward_atp_rate
        if katp:
            ATP = self.iap / (1 + 1 / katp)
        else:
            ATP = 0
        # compute the growth rate
        mu = scaled_phis["r"] * X * self.vt * ATP / (self.Katp + ATP)
        derivatives[self.X] = mu
        # summarize energy flows
        summary = """catabolic ATP flows:
{}

ATP sinks:
* maintenance: {:.2e}
* anabolism: {:.2e}

ATP/ADP: {:.2e} 

growth rate: {:.2e}""".format("\n".join("* {}: {:.2e}".format(pathway.name, flow/X) for pathway, flow in zip(self.pathways, cat_atp_flows)),
                            self.maintenance(T) / self.xaa / self.dGatp,
                            scaled_phis["r"] * self.vt * (self.nu_atp_tr + self.nu_atp_an),
                            katp,
                            mu)
        tracker.update_log(summary)
        # add proteome derivatives
        derivatives += self.proteome_derivatives(y, T, cat_atp_flows, katp, tracker)
        # print the rarest resources
        #rarity_pairs = ((name, dy / y) for name, y, dy in zip(self.chems_list, y, derivatives) if dy < 0)
        #sorted_pairs = sorted(rarity_pairs, key=lambda pair: abs(pair[1]), reverse=True)
        #rarest = "\n".join(f"{name:10.10}: {dy:.2e}" for name, dy in sorted_pairs[:5])
        #tracker.update_log(rarest)
        #tracker.update_log(f"pH: {-np.log10(y[self.chems_list.index('H+')]):.2f}")
        return derivatives

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
    def __init__(self, population_name, chems_list, reactions, params):
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
    - maintenance: maintenance energy flow rate (kJ.molX-1.hour-1)
    - energy barrier: total energy cost of biomass replication (dissipation + anabolism)
    - anabolism: name of the population's reaction which is the anabolic reaction
    - biomass: the specie to be considered as biomass in the anabolic equation
    (kJ.molX-1, negative)
    
    The following parameters must be defined for each pathway;
    - vmax: maximum catalytic rate
    - Km: affinity for substrate (dictionary)
    - norm: the chemical species by which the yield on the pathway is normalized
    (The "s" of the Yx/s. Usually the electron donor)
    """
    def __init__(self, population_name, chems_list, reactions, params):
        self.population_name = population_name
        self.chems_list = chems_list
        # take the specific parameters of the model
        self.X0 = params["X0"] # initial biomass concentration
        self.maintenance = params["maintenance"]
        self.energy_barrier = params["energy barrier"]
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
        for pathway in self.pathways:
            assert "Yxs" in pathway.parameters
            assert "norm" in pathway.parameters
            #pathway.normalize(pathway.parameters["norm"])
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
        self.anabolism.normalize(self.specific_chems["biomass"]["name"])

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

    def get_derivatives(self, y, T, tracker):
        X = y[self.X]
        rcat = np.hstack(self.FD.rate(pathway, y, T) * self.FT.rate(pathway, y, T)
                         for pathway in self.pathways)
        # rate of energy intake from the environment
        JG = sum(rcat_i * pathway.dG(y, T) for rcat_i, pathway in zip(rcat, self.pathways))
        ran = (JG - self.maintenance) / self.energy_barrier
        # each pathway is associated with a anabolism and catabolism stoichiometry
        metabolism = np.sum(np.array([rcat_i * pathway.stoichiometry_vector
                                      for rcat_i, pathway
                                      in zip(rcat, self.pathways)]), axis=0) + ran * self.anabolism.stoichiometry_vector
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

growth_models_dict = {"ThermoAllocationModel": ThermoAllocationModel,
                      "SimpleGrowthModel": SimpleGrowthModel,
                      "Stahl": Stahl,
                      }
