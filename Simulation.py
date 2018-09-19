from scipy.integrate import odeint, ode
import numpy as np
import pandas as pd
from Nesting import aggregate
from Spinner import spinner

class Simulation:
    def __init__(self, chems_list, nesting, system_equilibrator, system_glt, community,
    initial_concentrations, T, D, logger=lambda s, t, ay, ey: None):
        '''
        * chems_list: list of the name of the chemical species involved in the
        simulation, in the same order as in the expanded concentration vector
        * nesting: list of increasing integers indicating the way the chemical
        species are aggregated in the aggregated concentration vector (see the
        Nesting module)
        * system_equilibrator: SystemEquilibrator instance
        * community: Community instance
        * initial_concentration: aggregated concentration vector (numpy array)
        * T: system's temperature (1x1 float, K)
        * D: dilution rate (day-1)
        * logger: a function which is called at each integration step and which
        produces side effects intented for logging. The logger function is
        given the simulation instance, the current time, the aggregated
        concentrations vector and the expanded concentrations vector as
        arguments.
        '''
        self.chems_list = chems_list
        self.nesting = nesting
        self.system_equilibrator = system_equilibrator
        self.system_glt = system_glt
        self.community = community
        self.y0 = initial_concentrations
        # use the (equilibrated) initial concentrations as definition for the
        # chemostat input. Remove the biomassesm though
        self.input = np.array(self.equilibrate(self.y0))
        for population in self.community.populations:
            self.input[population.biomass_index] = 0
        self.T = T
        self.D = D
        self.logger = logger

    def equilibrate(self, y):
        '''Takes an aggregated concentration vector, returns an expanded
        concentration vector with all concentrations adjusted to account for
        the equilibria specified in system_equilibrator and the charge balance
        '''
        return self.system_equilibrator.equilibrate(y)

    def f(self, t, y):
        '''The function to integrate.
        * t: time (in day)
        * y: concentration vector (aggregated) (in M)
        '''
        # concentrations shall not be lower than the machine's epsilon
        y = np.clip(y, np.finfo(float).eps, None)
        expanded_y = np.array(self.equilibrate(y))
        # derivatives caused by biological reactions
        reaction_matrix = self.community.get_matrix(expanded_y, self.T)
        rates = self.community.get_rates(expanded_y, self.T)
        rate_matrix = np.diag(rates) @ reaction_matrix
        dy_dt_bio = np.sum(rate_matrix, axis=0) # expanded concentrations' derivatives
        # derivatives caused by the chemostat
        dy_dt_chemo = self.D * (self.input - expanded_y)
        # derivatives caused by gas-liquid transfers
        glt_rates = self.system_glt.get_rates(expanded_y, self.T)
        glt_matrix = self.system_glt.get_matrix()
        glt_rate_matrix = np.diag(glt_rates) @ glt_matrix
        dy_dt_glt = np.sum(glt_rate_matrix, axis=0)

        dy_dt = dy_dt_bio + dy_dt_chemo + dy_dt_glt
        return aggregate(dy_dt, self.nesting)

    def solve(self, time, dt):
        t = np.arange(0, time, dt)
        solver = ode(self.f)
        solver.set_integrator("lsoda")
        solver.set_initial_value(self.y0)

        ts = []
        ys = []

        while solver.successful() and solver.t < time:
            solver.integrate(solver.t + dt)
            expanded_y = self.equilibrate(solver.y)
            ts.append(solver.t)
            ys.append(expanded_y)
            self.logger(self, solver.t, solver.y, expanded_y)
            waitbar = spinner(int(solver.t), int(time))
            print("\rintegrating: {} day ".format(waitbar), end="")
        print()

        time_array = np.transpose(np.array(ts))
        conc_array = np.vstack(ys)

        data = np.column_stack((time_array, conc_array))
        labels = ["time"] + self.chems_list
        return pd.DataFrame(data=data, columns=labels)
