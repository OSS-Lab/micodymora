# Note to future reader:
# This module contains the functions parsing a simulation description and
# performing all the necessary precomputations to obtain a Simulation instance.
# For a better comprehension of the code, it is suggested to start reading the
# `get_simulation` function first

from copy import copy
import numpy as np
from Nesting import aggregate
from Chem import load_chems_dict
from Reaction import Reaction, MetabolicReaction, load_reactions_dict
from Equilibrium import SystemEquilibrator, load_equilibria_dict
from GLT import load_glt_dict, SimulationGasLiquidTransfer, SystemGasLiquidTransfers
from Rate import rates_dict
from Community import Community, Population
from Enzyme_allocation import enzyme_allocations_dict
from Constants import T0
from Simulation import Simulation

default_chems_description_path = "data/chems.csv"
default_reactions_description_path = "data/reactions.dat"
default_equilibria_description_path = "data/equilibria.dat"
default_glt_description_path = "data/gas_liquid_transfer.dat"

class OverlappingEquilibriaException(Exception):
    '''When the system's equilibria are defined, a single chemical species
    should not be affected by two different equilibria'''
    pass

# TODO: add an option to ignore species, for example water
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

def record_biomasses(chems_dict, chems_list, nesting, biomasses):
    '''For each population to simulate, creates an item in the chems list
    and create a new entry in the chem dictionary for the populations' biomass.
    Basically the new entry is a copy of an existing chem, but named after
    the population.
    * chems_dict: dictionary of Chem instances ({chem_name: Chem}) to be used
    * chems_list: list of the name of the chemical species in the system
    * biomasses: dictionary mapping the name of the populations in the system
       to the Chem instance used as a model for their biomass
    '''
    for population_name, biomass_model in biomasses.items():
       chems_list.append(population_name)
       population_biomass = copy(biomass_model)
       population_biomass.name = population_name
       chems_dict[population_name] = population_biomass
       nesting.append(nesting[-1] + 1)
    return chems_dict, chems_list, nesting

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
 
    catabolisms = set()
    biomasses = dict()
    for population_name, population_info in input_file.get("community", []).items():
        biomasses[population_name] = chems_dict[population_info["biomass"]["model"]]
        for catabolism in population_info["pathways"]:
            catabolisms.add(reactions_dict[catabolism])
 
    chems_list, nesting = inventory_chems(chems_dict, list(catabolisms), equilibria, glts, concentration_settings)
    chems_dict, chems_list, nesting = record_biomasses(chems_dict, chems_list, nesting, biomasses)
 
    return chems_list, nesting, equilibria, reactions_dict, chems_dict, glt_dict

def get_initial_concentrations(input_file, chems_list, nesting):
    '''Return the vector of initial concentrations in aggregated format.'''
    initial_concentrations = np.zeros(len(chems_list)) + np.finfo(float).eps
    concentration_settings = input_file.get("concentrations", {})
    for chem_name, chem_conc in concentration_settings.items():
        initial_concentrations[chems_list.index(chem_name)] = chem_conc

    # get concentration of biomass
    populations_concentrations = dict()
    for population_name, population_info in input_file.get("community", []).items():
        populations_concentrations[population_name] = population_info["biomass"]["concentration"]

    for pop_name, conc in populations_concentrations.items():
        initial_concentrations[chems_list.index(pop_name)] = chem_conc

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

def get_community(input_file, chems_dict, chems_list, reactions_dict):
    populations = list()
    for population_name, population_info in input_file.get("community").items():
        biomass_index = chems_list.index(population_name)
        catabolisms = list()
        for pathway in population_info["pathways"]:
            catabolism_info = population_info["pathways"][pathway]
            # instanciate rate function
            catabolism_rate = rates_dict[catabolism_info["rate"]](chems_list, catabolism_info["parameters"])
            # instanciate reaction
            reaction = reactions_dict[pathway]
            # instanciate the catabolism
            # population is set to None because it is not instanciated yet
            simulation_reaction = MetabolicReaction.from_reaction(reaction, chems_list, catabolism_rate)
            catabolisms.append(simulation_reaction)
        # enzyme allocation
        method_name = population_info["enzyme allocation"]["method"]
        parameters = population_info["enzyme allocation"]["parameters"]
        enzyme_allocation_function = enzyme_allocations_dict[method_name](parameters)
        # anabolism
        anabolism_info = population_info["anabolism"]
        anabolism_parameters = anabolism_info["parameters"]
        anabolism_rate = rates_dict[anabolism_info["rate"]](chems_list, anabolism_parameters)
        anabolism_reaction = MetabolicReaction.from_string(chems_dict, anabolism_info["reaction"], chems_list, anabolism_rate)
        populations.append(Population(population_name,
                                      catabolisms,
                                      enzyme_allocation_function,
                                      anabolism_reaction,
                                      biomass_index))
    return Community(populations)

def get_simulation(input_file):
    # get system's temperature
    # or assume it is the standard temperature if unspecified
    T = input_file.get("T", T0)
    # get the dilution rate or assume it is zero
    D = input_file.get("D", 0)
    # determine the list of chemical species which are involved in the
    # simulation
    chems_list, nesting, equilibria, reactions_dict, chems_dict, glt_dict = outline_systems_chemistry(input_file)
    y0 = get_initial_concentrations(input_file, chems_list, nesting)

    # instanciate the equilibrator
    chems_instances = [chems_dict[name] for name in chems_list]
    system_equilibrator = SystemEquilibrator(chems_instances, equilibria, nesting)

    # instanciate the gas-liquid transfers
    system_glt = get_system_glt(input_file, glt_dict, chems_list)
 
    # instanciate the community
    community = get_community(input_file, chems_dict, chems_list, reactions_dict)

    return Simulation(chems_list, nesting, system_equilibrator, system_glt, community, y0, T, D)
