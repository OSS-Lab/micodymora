# Note to future reader:
# This module contains the functions parsing a simulation description and
# performing all the necessary precomputations to obtain a Simulation instance.
# For a better comprehension of the code, it is suggested to start reading the
# `get_simulation` function first

from micodymora.Nesting import aggregate
from micodymora.Chem import Chem, load_chems_dict
from micodymora.Reaction import Reaction, MetabolicReaction, SimBioReaction, load_reactions_dict
from micodymora.Equilibrium import SystemEquilibrator, load_equilibria_dict
from micodymora.GLT import load_glt_dict, SimulationGasLiquidTransfer, SystemGasLiquidTransfers
from micodymora.Community import Community
from micodymora.Constants import T0
from micodymora.Simulation import Simulation
from micodymora.GrowthModel import growth_models_dict

from copy import copy
import numpy as np
import os
import re

module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
default_chems_description_path = os.path.join(module_path, "chems.csv")
default_dHf0_description_path = os.path.join(module_path, "dHf0.csv")
default_reactions_description_path = os.path.join(module_path, "reactions.dat")
default_equilibria_description_path = os.path.join(module_path, "equilibria.dat")
default_glt_description_path = os.path.join(module_path, "gas_liquid_transfer.dat")

_population_specific_chems_pat = re.compile("\{([^}]+)\}")

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
    total quantities rather than separately)
    
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
    Solving this problem implies to update the chems dictionary (because new chems are created
    to represent populations' biomass)
    '''
    # see if chems or reaction dictionary is overriden
    chems_description_path = input_file.get("chems_description_path", default_chems_description_path)
    dHf0_description_path = input_file.get("dHf0_description_path", default_dHf0_description_path)
    reactions_description_path = input_file.get("reactions_description_path", default_reactions_description_path)
    equilibria_description_path = input_file.get("equilibria_description_path", default_equilibria_description_path)
    glt_description_path = input_file.get("glt_description_path", default_glt_description_path)
    
    # load the chems and reactions dict
    chems_dict = load_chems_dict(chems_description_path, dHf0_description_path)
    reactions_dict = load_reactions_dict(chems_description_path, reactions_description_path, dHf0_path=default_dHf0_description_path)
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

    return chems_list, nesting, equilibria, chems_dict, glt_dict, populations

def get_catabolic_reactions(input_file, chems_dict, reactions_dict):
    '''This function collects the reactions in the config file in order to list
    the chems involved in the simulation. 
    '''
    for name, population in input_file.get("community").items():
        reactions = list()
        for reaction_name, parameters in population.get("pathways").items():
            if reaction_name in reactions_dict:
                reaction = reactions_dict[reaction_name]
            else:
                formula = parameters.get("formula")
                if formula:
                    # first look for population-specific variables and replace them by water
                    # because they should not be accounted for right now
                    formula = re.sub(_population_specific_chems_pat, "H2O", formula)
                    reaction = Reaction.from_string(chems_dict, formula, name=reaction_name)
                else:
                    raise ValueError("unknown reaction \"{}\" and no formula provided".format(reaction_name))
            reactions.append(reaction)
    return reactions

# This function parses the populations definitions in order to create the
# GrowthModel instances (which actually represent the populations).
def register_biomass(input_file, chems_dict, reactions_dict, chems_list, nesting):
    # temporary gas mask
    gas_species = np.array([chems_dict[species].phase == "g" and 1 or 0 for species in chems_list])
    population_instances = list()
    specific_reactions = dict() # {population name: {reaction name: {formula: str, parameters: list}}}
    specific_indexes = dict() # {population name: {generic chem name: index}}
    for population_name, population in input_file.get("community").items():
        parameters = population.get("growth model").get("parameters")
        generic_reactions = list()
        specific_reactions[population_name] = dict()
        # The reactions catalyzed by the population are scanned here.
        # Some are "generic", which means they only involve chems defined
        # a priori in the chems dictionary (like SO4-2, H2(aq) etc), while
        # other reactions are "specific", which mean they involve
        # population-specific chems, such as the population's biomass etc.
        # The specific reactions are set aside for the moment since the
        # specific chems are not registered yet. The population is instanciated
        # and given its generic reactions, then its specific chems are created,
        # plugged into the specific reactions, and the specific reactions given
        # to the population
        for reaction_name, reaction_parameters in population.get("pathways").items():
            is_a_generic_formula = True
            if reaction_name in reactions_dict:
                reaction = reactions_dict[reaction_name]
            else:
                formula = reaction_parameters.get("formula")
                if formula:
                    if _population_specific_chems_pat.findall(formula):
                        # If the formula refers to population-specific chems
                        # (such as the population's biomass) it cannot be
                        # instanciated right now
                        is_a_generic_formula = False
                        specific_reactions[population_name][reaction_name] = {"formula": formula,
                                                                              "parameters": reaction_parameters}
                    else:
                        reaction = Reaction.from_string(chems_dict, formula, name=reaction_name)
                else:
                    raise ValueError("unknown reaction \"{}\" and no formula provided".format(reaction_name))
            if is_a_generic_formula:
                simbioreaction = reaction.new_SimBioReaction(chems_list, gas_species, reaction_parameters)
                generic_reactions.append(simbioreaction)
        # instanciate the population
        growth_model_name = population.get("growth model").get("name")
        growth_model = growth_models_dict[growth_model_name](population_name, chems_list, generic_reactions, population.get("pathways"), parameters)
        # add the specific chems of the population to chems list, chems dict and nestings list
        # growth_model is expected to have a `specific_chems` attribute, which has the structure;
        # {local_name: {name: global_name, template: template}}
        # - local_name is the name used in specific reactions formulas (eg: "biomass")
        # - global_name is the name of the chem in the simulation (eg: "Dv total biomass")
        # - template is the name of the chem from chems_dict to use as a template
        # (eg: Biomass(Hoover)). template can be set to None.
        specific_chems = dict() # local name -> global name mapping for specific chems
        indexes = dict() # local name -> index in chems_list
        for chem_name, chem_info in growth_model.specific_chems.items():
            if chem_info["template"]:
                new_chem = chems_dict[chem_info["template"]].copy(name=chem_info["name"])
            else:
                new_chem = Chem(chem_name)
            if chem_info["name"] in chems_dict:
                raise ValueError("Specific chem \"{}\" of population \"{}\" overwrites already known chem".format(chem_info["name"], population_name))
            else:
                chems_dict[chem_info["name"]] = new_chem
            nesting.append(nesting[-1] + 1)
            chems_list.append(chem_info["name"])
            indexes[chem_name] = len(chems_list) - 1
            specific_chems[chem_name] = chem_info["name"]
        population_instances.append(growth_model)
        specific_indexes[population_name] = indexes
    # Now all populations have been scanned, the chems_list is complete
    # Give the knowledge of the chems_list to specific reactions and populations.
    # definitive gas mask
    gas_species = np.array([chems_dict[species].phase == "g" and 1 or 0 for species in chems_list])
    for population in population_instances:
        # update gas species vector for every pathway reactions
        for reaction in population.pathways:
            reaction.gas_species = gas_species
        # instanciate the specific reactions
        population_specific_reactions = list()
        for reaction_name, reaction_info in specific_reactions[population.population_name].items():
            # convert the local names in the specific reactions formulas into their global counterpart
            # (eg: convert "{biomass}" to "Dv_biomass" in the formula string)
            specific_chems_dict = {local_name: infos["name"] for local_name, infos in population.specific_chems.items()}
            formula = reaction_info["formula"].format(**specific_chems_dict)
            reaction = Reaction.from_string(chems_dict, formula, name=reaction_name)
            reaction = reaction.new_SimBioReaction(chems_list, gas_species, reaction_info["parameters"])
            population_specific_reactions.append(reaction)
        population.register_chems(specific_indexes[population.population_name], chems_list, population_specific_reactions)

    missing_enthalpies = set()
    for population in population_instances:
        for reaction in population.all_reactions:
            for reagent in reaction.reagents:
                dHf0 = chems_dict[reagent.name].dHf0
                if dHf0 is None:
                    missing_enthalpies |= {reagent.name}
    if missing_enthalpies:
        print("Cannot use correction of Gibbs energy for non-standard temperature: missing enthalpy data for the following species: {}".format(missing_enthalpies))
        print("Gibbs energy will be computed as dGr0 + RT ln Q")
    else:
        SimBioReaction.dG = SimBioReaction.dGT

    return chems_list, nesting, chems_dict, population_instances
        
def get_initial_concentrations(input_file, chems_list, nesting, populations):
    '''Return the vector of initial concentrations in aggregated format.'''
    initial_concentrations = np.zeros(len(chems_list))
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
    vliq = gl_info.get("vliq", 1)
    vgas = gl_info.get("vgas", 1)
    headspace_pressure = gl_info.get("headspace pressure", 1)
    alpha = gl_info.get("alpha", 0)
    if gl_info:
        glts = [SimulationGasLiquidTransfer.from_GasLiquidTransfer(glt_dict[glt], chems_list, kla, vliq, vgas)
                for glt, kla
                in (gl_info.get("transfers") or {}).items()]
    else:
        glts = []
    return SystemGasLiquidTransfers(chems_list, glts, vliq, vgas, headspace_pressure, alpha)

def get_simulation(input_file, logger=None, progress_tracker=None, check_uninitialized_sets=True):
    # get the absolute tolerance of the integrator
    atol = input_file.get("atol", None)
    # get system's temperature
    # or assume it is the standard temperature if unspecified
    T = input_file.get("T", T0)
    # get the dilution rate or assume it is zero
    D = input_file.get("D", 0)
    # get fixed pH, if pH is fixed
    fixed_pH = input_file.get("fixed pH", None)
    # determine the list of chemical species which are involved in the
    # simulation
    chems_list, nesting, equilibria, chems_dict, glt_dict, populations = outline_systems_chemistry(input_file)
    y0 = get_initial_concentrations(input_file, chems_list, nesting, populations)

    # instanciate the equilibrator
    chems_instances = [chems_dict[name] for name in chems_list]
    system_equilibrator = SystemEquilibrator(chems_instances, equilibria, nesting, fixed_pH=fixed_pH)

    # instanciate the gas-liquid transfers
    system_glt = get_system_glt(input_file, glt_dict, chems_list)
 
    # instanciate the community
    community = Community(populations)

    # check species sets initialized to null
    if check_uninitialized_sets:
        warning_issued = False
        equilibria = [{chem.name for chem in equilibrium.chems} for equilibrium in system_equilibrator.equilibria]
        transfers = [{transfer.liquid_chem.name, transfer.gas_chem.name} for transfer in system_glt.transfers]
        nests = list()
        seen = set()
        for equilibrium in equilibria:
            nest = equilibrium.copy()
            for transfer in transfers:
                if transfer & nest:
                    nest |= transfer
            if not nest & seen:
                nests.append(tuple(nest))
            seen |= nest
        chem_to_nest_mapping = {chem: [index for index, chems in enumerate(equilibria) if chem in chems] for chem in chems_list}
        nests_concentrations = [sum(y0[chem_to_nest_mapping[name][0]] for name in nest) for nest in nests]
        for n, c in zip(nests, nests_concentrations):
            if c == 0 and n != ("H+",) and n != ("HO-",):
                warning_issued = True
                print("WARNING: Set {} initialized with null concentration: this may cause numerical instability".format(n))
        if warning_issued:
            print("If null concentrations cause numerical unstability in the results, consider decreasing the \"atol\" parameter")

    # pass atol, logger and tracker if they are defined
    params = dict()
    if atol:
        params["atol"] = atol
    if logger:
        params["logger"] = logger
    if progress_tracker:
        params["progress_tracker"] = progress_tracker

    return Simulation(chems_list, nesting, system_equilibrator, system_glt, community, y0, T, D, **params)

if __name__ == "__main__":
    import yaml
    with open("simulation2.yaml", "r") as fh:
        input_file = yaml.safe_load(fh)
    get_simulation(input_file)
