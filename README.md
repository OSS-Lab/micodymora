# Micodymora user's guide

Micodymora is a python package allowing to simulate Ordinary Differential
Equations (ODE) models of microbial population dynamics, while providing
gas/liquid transfer and acide/base equilibria as additional features. It is
compatible with Python 3.4 and onward.

It contains an implementation of a simple microbial population dynamics model
(described in [this manuscript][1]), along with some
readily usable metabolic pathways and physico-chemical data (Henry constants,
acid/base equilibrium constants, Gibbs energy and enthalpies of formations),
plus a few convenience files to be able to run a simulation out of the box.

The way the package is organized makes it easy to add new data (reactions,
chemical species, acide/base equilibria, thermodynamics data...), so even users
without python programming skills can adapt the package to fit their needs. It
is also possible to implement new growth models and to simulate them using this
framework, although it will require python programming skills.

## dependencies

Besides standard library modules, the micodymora package relies on the
following modules;

- numpy

- scipy

- pandas

The utility script `utils/simulate.py` which allows to perform simulations out
of the box also requires the `PyYAML` module.

## structure of the project

The base directory of the project contains the code of the python package. The
`data/` directory contains the data for the package (definitions of chemical
species, chemical reactions, physicochemical constants etc...). The `utils/`
directory contains additional scripts which are not directly necessary for the
package but are provided for the users.

The whole project is intended to be stored in a directory called `micodymora/`
and to be used as a package by external scripts.

## Running a simulation using the Micodymora package

Micodymora is a python package, so in order to run a simulation, it must be
called by another python script. Conveniently, an example of such script is
provided (`utils/simulate.py`). It will also require a simulation configuration
file, which defines the experimental conditions of the simulation. The
`simulate.py` script reads configurations written as YAML files; an example of
such file is also provided with the project (`utils/DvMmMb_example.yaml`).

A minimal setup to run this example configuration would then be;

- make sure the project is stored inside a `micodymora/` directory

- step out of the directory and copy both `simulate.py` and
  `DvMmMb_example.yaml` there

- now the directory structure of the current directory should be

```
.
+-- simulate.py
+-- DvMmMb_example.yaml
+-- micodymora/
    +-- data/
    |   +-- (all the data files of the package ...)
    +-- utils/
    +-- (all the python files of the package ...)
```

The `simulate.py` script defines a Command Line Interface, so it allows to use
the micodymora from command line. As a first example, the following command
will simulate the system represented in the `DvMmMb_example.yaml` configuration
file for 24 hours and store the result in the current working directory.

> `python simulate.py DvMmMb_example.yaml --end 24`

Other possibilities of the `simulate.py` script are detailed in the
documentation of the script itself (try `python simulate.py --help`).

## micodymora's data

The package's data files are stored in the `data/` directory. It is possible to
add data to the package by editing those files.

### elements.dat

This file contains the atomic weight (g/mol) of the elements found in the
chemical species involved in the simulations. This is a "tab-separated values"
(tsv) file with a header. The elements names as they appear in this file must
be the same as they appear in the "formula" column of the `chems.csv` data
file.

### chems.csv

This file contains the definition of chemical species. This is a
"comma-separated values" (csv) file with a header. Every chemical species used
in the `reactions.dat`, `equilibria.dat` and `gas_liquid_transfer.dat` files
must be defined in `chems.csv`.

- The "name" column is the name by which the chemical species must be refered
  to in chemical equations (for example in the `reactions.dat` file).

- The "formula" column is the chemical formula of the chemical species. All the
  elements used in the formulas in this file must be defined in `elements.dat`.

- the "dGf0" column is the Gibbs energy of formation for the chemical species
  at 298 K in kJ/mol.

- the "source" column is the bibliography source for the dGf0 value. Obviously,
  this source value should not contain comma.

Note that for any given chemical species, there must be one entry for each
state into which the chemical species can be found in the simulations. For
example, in  `chems.csv` there is an entry for "H2(aq)" (dihydrogen dissolved
in the liquid phase) and an entry for "H2(g)" (dihydrogen in the gas phase).
It is so because in many circumstances, we may want to simulate them as
independent species, and they have different Gibbs energies and enthalpies.
If a chemical species belongs to a specific phase, it must be indicated with
the name of the phase in parentheses. If no phase is specified in a chemical
species name, it is assumed to be dissolved in the liquid phase.

### dHf0.csv

This file contains the enthalpy of formation of chemical species at 298 K in
kJ/mol. It is not necessary that all species have an enthalpy value defined in
this file in order to run a simulation. However, if some enthalpy values for
chemical species involved in a simulation are missing, the program will not be
able to apply temperature correction on Gibbs energies, so the standard
condition of 298 K will be assumed (a message will indicate this at the
beginning of the simulation).

### reactions.dat

This file contains chemical reaction definitions. This is not a list of all the
simulations occuring in a given simulation, but a list of predefined reactions
from which the user can choose from. It is not necessary that all chemical
reactions are defined in that file; they can be defined inside the
configuration file, however this file allow to provide shorthands by assigning
a name to predefined reactions.

Each line of this file correspond to a chemical reaction. It begins with the
reaction's name (by which it will be refered to in simulation configuration
files), then a column symbol, and then the reaction's formula.

Chemical species in the formula must be referred to by their *name* as it is
defined in the `chems.csv` file. Their stoichiometric coefficient must be
placed at the beginning of their name without space (for example "2 H2O" is
wrong, while "2H2O" is right). The chemical species names must be separated
with "+" and "-->" (arrow) symbols, themselves being separated by spaces.
For example, the following line

> `acetogenesis: Lac(-) + 2H2O --> Ac(-) + 2H2(aq) + HCO3- + H+`

is a valid reaction definition while

> `acetogenesis: Lac(-)+2H2O-->Ac(-)+2H2(aq)+HCO3-+H+`

is not.

### equilibria.dat

This file contains the definition of acide/base equilibria which are considered
to apply instantly in the reaction medium. Like `reactions.dat`, the program's
user picks which ones applies to a specific simulate by refering to their names
in the simulation configuration files.

Each line defines a specific chain of equilibria. There are three fields
separated by ": ". The first field is the name of the equilibria chain. The
second field is a space-separated list of the chemical species involved in the
equilibria chain (again, the names must be defined in `chems.csv`), and the
third field contains the pKa values of the equilibria at 298 K.

For example the following row;

> `carbonate: CO2(aq) HCO3- CO3-2: 6.35 10.33`

defines an equilibria chain called "carbonate", which consists in two
equilibria; "CO2(aq) <-> HCO3-" and "HCO3- <-> CO3-2", whose pKa constants at
298 K are respectively 6.35 and 10.33.

In its current state, micodymora can only account for linear, independent
equilibria chains. For example, multiple equilibria between free cations and
CO3-2 (CO3-2 + Mg+2 <-> MgCO3, CO3-2 + Ca+2 <-> CaCO3 ...) cannot be handled
for the moment. If the acid/base equilibria selected in a configuration file
are such that a single chemical species is affected by multiple equilibria, an
error will be raised during the parsing of the configuration file.

### gas_liquid_transfer.csv

This file contains the definition of transfers of chemical species between the
gas phase and the liquid phase. This is a "comma-separated values" (csv) file
with a header. Like `reactions.dat`, the program's user picks which ones
applies to a specific simulate by refering to their names in the simulation
configuration files. Column "name" defines the name of the gas/liquid transfer
as it should be referred to in configuration files. Columns "liquid" and "gas"
contains the name of the chemical species in liquid and gas phase respectively.
Those names must be the name of species already defined in `chems.csv`. Column
H0cp contains the value of the Henry constant for the transfer between the two
phases, in mol per meter cube per Pascal. This constant, when multiplied to a
gas pressure in Pascal, then gives the saturation concentration of the liquid
phase exposed to the gas phase at 298 K. The "Hsol/R" column contains the
enthalpy of solution of the gas/liquid transfer, divided by the gas constant R.
The value of this column is then in Kelvin. This value is needed in order to
update the Henry constant of the gas/liquid transfer for temperatures different
from 298 K.

## writing a simulation configuration file

A configuration file contains the informations necessary for the micodymora
simulation functions to be able to represent a microbial culture as an ordinary
differential equations system and to integrate it.

The utility files provided in the `utils/` directory rely on the YAML format,
however, the input required by the `SimulationDescriptionParser.py`
`get_simulation` functions is only a python dictionary, so it is easy to write
a script to read data from other sources such as JSON files, or even to
directly generate configuration data structures in python.

### mandatory parameters

- `concentrations`: a dictionary defining the initial concentrations of
  chemical species in the system, where the entries are the name of chemical
  species as defined in `data/chems.csv` and the values are their concentration
  in mole per liter. If the species is a gas species, its concentration is
  interpreted as being in mole per liter of gas phase.

- `community`: a dictionary of the microbial community to simulate. Each entry
  is the name of a microbial population (which will be attributed a biomass
  concentration), and it contains another dictionary with a "pathways" entry and
  a "growth model" entry. The "pathways" entry contains a dictionary, whose
  entries are reaction names, which *can* refer to readily defined reactions in
  `data/reactions.dat`. It is also possible to use instead a name which is not
  defined in `data/reactions.dat` and instead write the formula of the reaction
  in a "formula" entry inside the reaction's dictionary. The dictionary of each
  reaction must also contain reaction-specific parameters, such as maximum rate
  etc, which depends on the growth model. The "growth model" entry of the
  population dictionary also contains a dictionary, which contains a "name"
  entry indicating the name of the growth model to use (refer to
  `GrowthModel.py`) and a "paramters" entry which contains population-specific
  parameters for the model (which depends on the growth model).

Here is a summary of the "community" data structure in YAML format, with the
syntactic names in uppercase;

```
community:
    POPULATION NAME:
        pathways:
            REACTION 1 NAME: # this reaction name can be found in data/reactions.dat
                PARAMETER 1: PARAMETER 1 VALUE
                PARAMETER 2: PARAMETER 2 VALUE
            REACTION 2 NAME: # this reaction name cannot be found in data/reactions.dat
                formula: REACTION FORMULA # use the same convention as in data/reactions.dat
                PARAMETER 1: PARAMETER 1 VALUE
                PARAMETER 2: PARAMETER 2 VALUE
        growth model:
            name: MODEL NAME
            parameters:
                POPULATION PARAMETER 1: PARAMETER 1 VALUE
                POPULATION PARAMETER 2: PARAMETER 2 VALUE
```

### optional parameters

- `atol`: the absolute tolerance for the integrator (default: 1e-3)

- `T`: the temperature of the system (default: 298.15 K)

- `D`: the dilution rate occuring in the system, in order to simulate a
  chemostat culture (default: 0 hour-1)

- `fixed pH`: a constant pH value for the system. If this parameter is defined,
  the pH of the pH will be constant and all the acid/base equilibria will be
  computed during each timesteps so as to comply to this pH value (default:
  None)

- `equilibria`: list of acid/base equilibria, by their name as they are defined
  in `data/equilibria.dat`

- `gas-liquid interface`: informations to provide in the case of a culture
  medium having a gas phase;

- - vliq: volume of the liquid phase in liter (default: 1 L)

- - vgas: volume of the gas phase in liter (default: 1 L)

- - transfers: dictionary of the different gas/liquid transfers happening
  during the simulation. The key of each dictionary entry is the name of the
  name of a gas/liquid transfer as defined in `gas_liquid_transfer.csv` and
  the value of each entry is its mass transfer coefficient (kLa) in hour-1 (cf
  Lewis and Whitman, 1924).

[1]: https://biorxiv.org/cgi/content/short/857276v1
