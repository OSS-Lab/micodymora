from micodymora.Constants import Rkj, T0, F

import abc
import numpy as np
import functools
import operator

# TODO: test this module
# TODO: find a way to build those structures with the parser

class GrowthModel(abc.ABC):
    '''Classes inheriting from GrowthModel represent a microbial population,
    and define its growth kinetics.

    Required attributes:
    * specific_chems: name of the specific chems for an instanciated population.
    * affected_by_dilution: boolean vector specifying whether the specific
    chems of the population are affected by the chemostat's dilution rate or not
    '''
    def __init__(self, population_name, chems_list, reactions, params):
        self.specific_chems = NotImplemented
        self.affected_by_dilution = NotImplemented
        super().__init__()

    @abc.abstractmethod
    def get_derivatives(self, expanded_y, T, tracker):
        pass

    @abc.abstractmethod
    def register_chems(self, indexes):
        '''This message is supposed to be sent by the configuration parser,
        during the building of the simulation. At that point, the specific
        chems introduced by the growth model have been added to the
        concentration vector of the simulation, so the parser sends their
        index back to the GrowthModel instance through this message
        * indexes: list of the index of each population-specific chem in
        the concentration vector of the simulation.
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
        self.specific_chems = ["{} total biomass".format(self.population_name),
                               "{} phi r".format(self.population_name)]
        self.specific_chems += ["{} phi cat {}".format(self.population_name, pathway.name)
                                for pathway
                                in self.pathways]
        # only total proteome and atp should be subjected to dilution
        self.affected_by_dilution = [False for chem in self.specific_chems]
        self.affected_by_dilution[0] = True # total proteome

    def register_chems(self, indexes):
        # The type of chem to which the indexes correspond is set by the order
        # of the chems in self.specific_chems
        self.X = indexes[0] # chem 1 is for total biomass
        self.phi_r = indexes[1] # chem 2 is for ribosome fraction
        # following chems are for pathway-specific enzyme fractions, in the same order
        # as in self.pathway
        self.phi_cat = {pathway.name: phi for pathway, phi in zip(self.pathways, indexes[2:])}
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
        cat = {name: y[phi_cat] / total_phi * (1 - self.phi_x - self.phi_an)
               for name, phi_cat 
               in self.phi_cat.items()}
        other = {"x": self.phi_x, "r": max(0, y[self.phi_r]) / total_phi * (1 - self.phi_x - self.phi_an)}
        return {**other, **cat}

    def proteome_derivatives(self, y, T, cat_atp_flows, katp):
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
            for population, ideal_phi_cat in zip(self.pathways, ideal_phi_cats):
                phi_cat = y[self.phi_cat[population.name]]
                derivatives_vector[self.phi_cat[population.name]] = self.k * (ideal_phi_cat - phi_cat)

        # phi r
        derivatives_vector[self.phi_r] = self.k * y[self.phi_r] * (katp - self.iar)

        return derivatives_vector

    def rcat(self, y, T):
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
        rcat = self.rcat(y, T)
        X = y[self.X]
        derivatives = rcat @ self.reaction_matrix
        # compute the ATP concentration
        cat_atp_flows = rcat * self.nu_atp_cat
        forward_atp_rate = sum(cat_atp_flows)
        backward_atp_rate = X * (self.maintenance(T) / self.xaa / self.dGatp
                                 + scaled_phis["r"] * self.vt * (self.nu_atp_tr + self.nu_atp_an))
        katp = forward_atp_rate / backward_atp_rate
        tracker.update_log(f"{self.population_name}: {katp:.2e}")
        if katp:
            ATP = self.iap / (1 + 1 / katp)
        else:
            ATP = 0
        # compute the growth rate
        mu = scaled_phis["r"] * X * self.vt * ATP / (self.Katp + ATP)
        derivatives[self.X] = mu
        # add proteome derivatives
        derivatives += self.proteome_derivatives(y, T, cat_atp_flows, katp)
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
               "JinBethkeFT": JinBethkeFT,
               "LaRowe2012FT": LaRowe2012FT}

growth_models_dict = {"ThermoAllocationModel": ThermoAllocationModel}
