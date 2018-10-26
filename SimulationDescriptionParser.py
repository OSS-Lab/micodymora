# Note to future reader:
# This module contains the functions parsing a simulation description and
# performing all the necessary precomputations to obtain a Simulation instance.
# For a better comprehension of the code, it is suggested to start reading the
# `get_simulation` function first

from micodymora.Nesting import aggregate
from micodymora.Chem import Chem, load_chems_dict
from micodymora.Reaction import Reaction, MetabolicReaction, load_reactions_dict
from micodymora.Equilibrium import SystemEquilibrator, load_equilibria_dict
from micodymora.GLT import load_glt_dict, SimulationGasLiquidTransfer, SystemGasLiquidTransfers
from micodymora.Community import Community
from micodymora.Constants import T0
from micodymora.Simulation import Simulation
from micodymora.GrowthModel import growth_models_dict

from copy import copy
import numpy as np
import os

module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
default_chems_description_path = os.path.join(module_path, "chems.csv")
default_reactions_description_path = os.path.join(module_path, "reactions.dat")
default_equilibria_description_path = os.path.join(module_path, "equilibria.dat")
default_glt_description_path = os.path.join(module_path, "gas_liquid_transfer.dat")

class OverlappingEquilibriaException(Exception):
    '''When the system's equilibria are defined, a single chemical species
    should not be affected by two different equilibria'''
    pass

def inventory_chems(chems_dict, reactions, equilibria, glts, initial_concentrations, nevermind=["H2O"]):
    '''Inventories the chemical species needed in a simulation,
    create the population-specific biomass chems,
    set the the order of the chems in the concentration vector,
    output it as a list of strings.
    Also determines the nesting of the different chems (how they group
    according to their equilibrium, so the groups can be simulated together as
    total quantities rahter than separately)
    
    * chems_dict: dictionary of Chem instances ({chem_name: Chem}) to be used
    * reactions: list of Reaction instances representing all the catabolic reactions
    * equilibria: list of Equilibrium instances representing the equilibria to consider
    * glts: list of GasLiquidTransfer instances
    * initial concentration: dictionary ({Chem: concentration}) assigning initial
            concentrations (it does not have to be exhaustive)
    * nevermind: list of names of chemical species to ignore during the simulation
    '''
    chems_list = ["H+", "HO-"]
    nesting = [0, 1, 2]
    for equilibrium in equilibria:
        for chem in equilibrium.chems:
            if chem.name in chems_list:
                raise OverlappingEquilibriaException("{} affected by multiple equilibria".format(chem.name))
            elif chem.name not in nevermind:
                chems_list.append(chem.name)
        nesting.append(nesting[-1] + len(equilibrium.chems))
    for glt in glts:
        liquid = glt.liquid_chem.name
        gas = glt.gas_chem.name
        if liquid not in chems_list and liquid not in nevermind:
            chems_list.append(liquid)
            nesting.append(nesting[-1] + 1)
        if gas not in chems_list and gas not in nevermind:
            chems_list.append(gas)
            nesting.append(nesting[-1] + 1)
    for reaction in reactions:
        for chem in reaction.reagents:
            if chem.name not in chems_list and chem.name not in nevermind:
                chems_list.append(chem.name)
                nesting.append(nesting[-1] + 1)
    for chem in initial_concentrations:
        if chem not in chems_list and chem not in nevermind:
            chems_list.append(chem)
            nesting.append(nesting[-1] + 1)
    return chems_list, nesting

def outline_systems_chemistry(input_file):
    '''This function does a lot of different things. Basically, it looks at all
    the chemical reactions and equilibria that are supposed to happen in the
    system, and it decides which chemical specie shall be present in the
    concentration vector of the simulation, and in which order.
    Solving this problem implies to update the chems dictionary (because new
    chems are created to represent populations' biomass).
    '''
    # see if chems or reaction dictionary is overriden
    chems_description_path = input_file.get("chems_description_path", default_chems_description_path)
    reactions_description_path = input_file.get("reactions_description_path", default_reactions_description_path)
    equilibria_description_path = input_file.get("equilibria_description_path", default_equilibria_description_path)
    glt_description_path = input_file.get("glt_description_path", default_glt_description_path)
    
    # load the chems and reactions dict
    chems_dict = load_chems_dict(chems_description_path)
    reactions_dict = load_reactions_dict(chems_description_path, reactions_description_path)
    equilibria_dict = load_equilibria_dict(chems_description_path, equilibria_description_path)
    glt_dict = load_glt_dict(chems_description_path, glt_description_path)
 
    # look for concentration settings, catabolisms and equilibria to find which
    # chemical species should be simulated
    concentration_settings = input_file.get("concentrations", {})
    equilibria = [equilibria_dict[equilibrium_name]
                  for equilibrium_name
                  in (input_file.get("equilibria") or [])]

    gl_info = input_file.get("gas-liquid interface")
    if gl_info:
        glts = [glt_dict[glt]
                for glt
                in gl_info.get("transfers", [])]
    else:
        glts = []

    # look at the pathways for catabolic reagents, without instanciating the populations
    reactions = get_catabolic_reactions(input_file, chems_dict, reactions_dict)
    # instanciate a preliminary chems list, without the growth-model-specific variables
    preliminary_chems_list, nesting = inventory_chems(chems_dict, reactions, equilibria, glts, concentration_settings)
    # instanciate the populations and add their specific variables to the chems list
    chems_list, nesting, chems_dict, populations = register_biomass(input_file, chems_dict, reactions_dict, preliminary_chems_list, nesting)

    return chems_list, nesting, equilibria, reactions_dict, chems_dict, glt_dict, populations

def get_catabolic_reactions(input_file, chems_dict, reactions_dict):
    for name, population in input_file.get("community").items():
        reactions = list()
        for reaction_string in population.get("pathways"):
            if "-->" in reaction_string:
                reaction = Reaction.from_string(chems_dict, reaction_string)
            else:
                reaction = reactions_dict[reaction_string]
            reactions.append(reaction)
    return reactions

def register_biomass(input_file, chems_dict, reactions_dict, chems_list, nesting):
    '''
    '''
    population_instances = list()
    for name, population in input_file.get("community").items():
        # create the growth model instances for each population
        # firstly they are created but they lack the knowledge
        # of the index of their specific variables (biomass, atp...) as it is
        # not yet recorded in the chems list
        # FIXME: maybe this part and get_catabolic_reactions could be factorized
        parameters = population.get("growth model").get("parameters")
        reactions = list()
        for reaction_string, reaction_parameters in population.get("pathways").items():
            if "-->" in reaction_string:
                reaction = Reaction.from_string(chems_dict, reaction_string)
            else:
                reaction = reactions_dict[reaction_string]
            simbioreaction = reaction.new_SimBioReaction(chems_list, reaction_parameters)
            reactions.append(simbioreaction)
        # instanciate the population
        growth_model_name = population.get("growth model").get("name")
        growth_model = growth_models_dict[growth_model_name](name, chems_list, reactions, parameters)
        population_instances.append(growth_model)
        # add the specific chems of the population to chems list, chems dict and nestings list
        chems_list.extend(growth_model.specific_chems)
        for chem in growth_model.specific_chems:
            nesting.append(nesting[-1] + 1)
            chems_dict[chem] = Chem(chem)

    # give the populations the knowledge of the index of their specific variables
    for population in population_instances:
        specific_chems_indexes = [chems_list.index(chem) for chem in population.specific_chems]
        population.register_chems(specific_chems_indexes)

    # record the growth-model-specific chems
#    for chem in growth_model.specific_chems:
#        nesting.append(nesting[-1] + 1)
#        specific_chems_indexes.append(len(chems_list) - 1)
#        chems_dict[chem] = Chem(chem)
#    growth_model.register_chems(specific_chems_indexes)

        # now the populations have the knowledge of the index of their specific
        # variables
    return chems_list, nesting, chems_dict, population_instances

def get_initial_concentrations(input_file, chems_list, nesting, populations):
    '''Return the vector of initial concentrations in aggregated format.'''
    initial_concentrations = np.zeros(len(chems_list)) + np.finfo(float).eps
    concentration_settings = input_file.get("concentrations", {})
    for chem_name, chem_conc in concentration_settings.items():
        initial_concentrations[chems_list.index(chem_name)] = chem_conc

    # set concentration of populations-specific variables
    for population in populations:
        initial_concentrations = population.set_initial_concentrations(initial_concentrations)

    return aggregate(initial_concentrations, nesting)

def get_system_glt(input_file, glt_dict, chems_list):
    # get list of SimulationGasLiquidTransfer instances
    gl_info = input_file.get("gas-liquid interface")
    if gl_info:
        glts = [SimulationGasLiquidTransfer.from_GasLiquidTransfer(glt_dict[glt], chems_list, kla)
                for glt, kla
                in (gl_info.get("transfers") or {}).items()]
    else:
        glts = []
    vliq = gl_info.get("vliq", 1)
    vgas = gl_info.get("vgas", 1)
    headspace_pressure = gl_info.get("headspace pressure", 1)
    alpha = gl_info.get("alpha", 1)
    return SystemGasLiquidTransfers(chems_list, glts, vliq, vgas, headspace_pressure, alpha)

def get_simulation(input_file, logger=None):
    # get system's temperature
    # or assume it is the standard temperature if unspecified
    T = input_file.get("T", T0)
    # get the dilution rate or assume it is zero
    D = input_file.get("D", 0)
    # determine the list of chemical species which are involved in the
    # simulation
    chems_list, nesting, equilibria, reactions_dict, chems_dict, glt_dict, populations = outline_systems_chemistry(input_file)
    y0 = get_initial_concentrations(input_file, chems_list, nesting, populations)

    # instanciate the equilibrator
    chems_instances = [chems_dict[name] for name in chems_list]
    system_equilibrator = SystemEquilibrator(chems_instances, equilibria, nesting)

    # instanciate the gas-liquid transfers
    system_glt = get_system_glt(input_file, glt_dict, chems_list)
 
    # instanciate the community
    community = Community(populations)

    # pass a logger if defined
    if logger:
        params = {"logger": logger}
    else:
        params = {}

    return Simulation(chems_list, nesting, system_equilibrator, system_glt, community, y0, T, D, **params)

if __name__ == "__main__":
    import yaml
    with open("simulation2.yaml", "r") as fh:
        input_file = yaml.safe_load(fh)
    get_simulation(input_file)
