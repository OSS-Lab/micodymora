"""This script allows to use the micodymora package to perform simulations
from command-line. To use it, you need to provide a valid yaml configuration
file.

If the simulation is a success, the simulated concentrations will be written in
a log file. The default location of the log file is the current working
directory, another location can be specified using the --log option.  """

import numpy as np
import argparse
import datetime
import yaml
import os
import micodymora.SimulationDescriptionParser
from micodymora.Simulation import simulation_status, AbstractLogger, BlankProgressTracker

# functions to check the validity of the arguments
def path_to_config(path):
    """Check that the simulation config path is valid (ie it points to an
    existing file and the user has reading permission on it)"""
    if os.path.isfile(path):
        if os.access(path, os.R_OK):
            return path
        else:
            raise argparse.ArgumentTypeError("you do not have reading permission on this file")
    else:
        raise argparse.ArgumentTypeError("this is not a valid path to a file")

def path_to_log_dir(path):
    """Check that the simulation log path is valid (ie it points to an existing
    directory and the user has writting permission on it)"""
    if os.path.isdir(path):
        if os.access(path, os.W_OK):
            return path
        else:
            raise argparse.ArgumentTypeError("you do not have writting permission on this directory")
    else:
        raise argparse.ArgumentTypeError("this is not a valid path to a directory")

# definition of the arguments parser
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("config", help="configuration file", type=path_to_config)
parser.add_argument("-t", "--timepoints", help="file containing the timepoints (numbers separated by newlines)")
parser.add_argument("-e", "--end", help="Number of hours to simulate. Ignored if -t is specified.", type=float)
parser.add_argument("-p", "--points", help="number of rows to log. Ignored if -t is specified", type=int, default=15)
parser.add_argument("-l", "--log", help="log directory", type=path_to_log_dir, default=os.path.dirname(os.path.realpath(__file__)))
parser.add_argument("-n", "--log-name", help="name of the concentration log file", default="concentrations.csv")
parser.add_argument("--mode", help="integration mode", choices=["ivp", "by-chunk", "odeint"], default="ivp")
parser.add_argument("--method", help="integration method (relevant only if mode is ivp)", choices=["RK45", "LSODA", "RK23", "Radau", "BDF"], default="RK45")

# parsing of the arguments
args = parser.parse_args()
with open(args.config, "r") as stream:
    # Read the configuration file with python's YAML parser
    try:
        simulation_data = yaml.safe_load(stream)
    except Exception as e:
        # If it fails here, check YAML format specifications.
        # indentations must be spaces, and numbers written with powers of ten
        # must have a decimal (for example 1e-3 will raise an error while
        # 1.0e-3 does not)
        print("{} does not contain proper yaml".format(args.config))
        raise e

    # Read the data structure with micodymora's configuration parser
    try:
        simulation = micodymora.SimulationDescriptionParser.get_simulation(simulation_data)
    except Exception as e:
        # If it fails here, it means the informations themselves are not what
        # micodymora's parser expects.
        print("{} is not a valid simulation file".format(args.config))

# if neither a timepoints file or an end time is specified, throw an error
if args.timepoints is None and args.end is None:
    raise argparse.ArgumentError("Either --timepoints or --end must be provided")

if args.timepoints:
    try:
        with open(args.timepoints, "r") as stream:
            # WARNING: timepoints are automatically sorted and uniqued
            timepoints_list = sorted({float(timepoint) for timepoint in stream.readlines()})
            if timepoints_list[0] != 0:
                timepoints_list.insert(0, 0.0)
            timepoints = np.asarray(timepoints_list)

    except Exception as e:
        print("{} is not a valid timepoints file".format(args.timepoints))
        raise e
else:
    timepoints = np.linspace(0, args.end, args.points)

# simulation
start_time = datetime.datetime.now()
result = simulation.solve(timepoints, mode=args.mode, method=args.method)

# post-simulation
end_time = datetime.datetime.now()
concentrations_log_file = os.path.join(args.log, args.log_name)
result.to_csv(concentrations_log_file, index=False)
simulation_status_meaning = next(meaning for meaning, code in simulation_status.items() if code == simulation.status)
print("simulation ended with status {} ({})".format(simulation.status, simulation_status_meaning ))
print("Elapsed time: {}".format(end_time - start_time))
print("Simulation results written to {}".format(concentrations_log_file))
