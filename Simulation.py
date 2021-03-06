from micodymora.Nesting import aggregate
from micodymora.Spinner import spinner

from scipy.integrate import ode, odeint, solve_ivp
import abc
import numpy as np
import pandas as pd

class AbstractLogger(abc.ABC):
    '''Defines the interface that a logger class should have'''
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def do_log(self, simulation, time, nested_y, expanded_y):
        pass

class BlankLogger(AbstractLogger):
    '''Minimalistic implementation of a Logger class.
    Basically does nothing.'''
    def __init__(self):
        pass

    def do_log(self, simulation, time, nested_y, expanded_y):
        pass

class AbstractProgressTracker(abc.ABC):
    '''Define the interface that an instance tracking simulation
    progress in real time should have'''
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def set_total_time(self, total_time):
        pass

    @abc.abstractmethod
    def update_progress(self, time):
        '''
        * time: current simulation time
        '''
        pass

    @abc.abstractmethod
    def update_log(self, message):
        '''
        * message: string
        '''
        pass

class BlankProgressTracker(AbstractProgressTracker):
    def __init__(self):
        pass

    def set_total_time(self, total_time):
        self.total_time = total_time

    def update_progress(self, time):
        pass

    def update_log(self, message):
        pass

simulation_status = {
"has not run yet": 0,
"is running": 1,
"has run until the end": 2,
"has run but stopped before the end": 3}
class Simulation:
    def __init__(self, chems_list, nesting, system_equilibrator, system_glt, community,
    initial_concentrations, T, D, atol=1e-3, logger=BlankLogger(), progress_tracker=BlankProgressTracker()):
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
        * D: dilution rate (hour-1)
        * atol: absolute tolerance of the integrator
        * logger: a logger instance inheriting from AbstractLogger
        '''
        self.atol = atol
        self.endpoint = None # end time for integration
        self.t = None # current point of the integration
        self.chems_list = chems_list
        self.nesting = nesting
        self.system_equilibrator = system_equilibrator
        self.system_glt = system_glt
        self.community = community
        self.y0 = initial_concentrations
        self.nested_species_names = [str(equilibrium).replace("eq: ", "") for equilibrium in self.system_equilibrator.equilibria]
        # use the (equilibrated) initial concentrations as definition for the
        # chemostat input. Remove the population-specific variables
        # (ATP, biomass...).
        self.input = np.array(self.equilibrate(self.y0))
        for index in self.community.get_population_specific_variables_indexes():
            self.input[index] = 0
        self.T = T
        self.D = D
        # prepare a vector of dilution rates, where the dilution rate applyied to
        # each chem is either D or 0 (D being the global dilution rate defined for
        # the culture medium)
        self.D_vector = np.ones(len(chems_list)) * D
        for index in self.community.get_index_of_chems_unaffected_by_dilution():
            self.D_vector[index] = 0
        assert issubclass(type(logger), AbstractLogger)
        self.logger = logger
        assert issubclass(type(progress_tracker), AbstractProgressTracker)
        self.progress_tracker = progress_tracker
        self.status = simulation_status["has not run yet"]

    def equilibrate(self, y):
        '''Takes an aggregated concentration vector, returns an expanded
        concentration vector with all concentrations adjusted to account for
        the equilibria specified in system_equilibrator and the charge balance
        '''
        return self.system_equilibrator.equilibrate(y)

    def concentrations_pretreatment(self, y):
        '''This function centralizes the eventual pretreatments performed
        on the concentration vector y before derivatives are computed
        based on it.
        The purpose of this function is to make those modifications
        more explicit and to provide the eventual Logger instance with
        an easy way to call them.'''
        # concentrations shall not be lower than the machine's epsilon
        return np.clip(y, np.finfo(float).eps, None)

    def f(self, t, y):
        '''The function to integrate.
        * t: time (in hour)
        * y: concentration vector (aggregated) (in M)
        '''
        self.t = t
        print("\rintegrating: {} hour ".format(spinner(int(t), int(self.endpoint))), end="")
        y = np.clip(y, 0, None)
        expanded_y = np.array(self.equilibrate(y))
        # derivatives caused by biological reactions
        bio_derivatives = self.community.get_derivatives(expanded_y, self.T, self.progress_tracker)
        dy_dt_bio = np.sum(bio_derivatives, axis=0)
        # derivatives caused by the chemostat
        dy_dt_chemo = self.D_vector * (self.input - expanded_y)
        # derivatives caused by gas-liquid transfers
        glt_rates = self.system_glt.get_rates(expanded_y, self.T, self.progress_tracker)
        glt_matrix = self.system_glt.get_matrix()
        glt_rate_matrix = np.matmul(np.diag(glt_rates), glt_matrix)
        dy_dt_glt = np.sum(glt_rate_matrix, axis=0)
        dy_dt = dy_dt_bio + dy_dt_chemo + dy_dt_glt
        return aggregate(dy_dt, self.nesting)

    # This is supposed to be an exact copy of the "normal" f function
    # except that the arguments are inverted (y before t), so as to
    # satisfy the signature requirements of odeint. Wrapping the f
    # function would have seemed more sensible than writing this copy,
    # however the only reason why odeint integration is kept alongside
    # other integration modes is speed, while wrapping f would have
    # increased the number of calls and thus decreased the interest
    # of keeping odeint as an option.
    def f_for_odeint(self, y, t):
        '''The function to integrate.
        * t: time (in hour)
        * y: concentration vector (aggregated) (in M)
        '''
        self.t = t
        print("\rintegrating: {} hour ".format(spinner(int(t), int(self.endpoint))), end="")
        y = np.clip(y, 0, None)
        expanded_y = np.array(self.equilibrate(y))
        # derivatives caused by biological reactions
        bio_derivatives = self.community.get_derivatives(expanded_y, self.T, self.progress_tracker)
        dy_dt_bio = np.sum(bio_derivatives, axis=0)
        # derivatives caused by the chemostat
        dy_dt_chemo = self.D_vector * (self.input - expanded_y)
        # derivatives caused by gas-liquid transfers
        glt_rates = self.system_glt.get_rates(expanded_y, self.T, self.progress_tracker)
        glt_matrix = self.system_glt.get_matrix()
        glt_rate_matrix = np.matmul(np.diag(glt_rates), glt_matrix)
        dy_dt_glt = np.sum(glt_rate_matrix, axis=0)
        dy_dt = dy_dt_bio + dy_dt_chemo + dy_dt_glt
        return aggregate(dy_dt, self.nesting)

    def f_diagnosis(self, t, y):
        '''Do the same thing as the f function except that it does not
        reaggregate the variables. It is used to show the derivatives
        in the system after the integration has failed
        '''
        expanded_y = np.array(self.equilibrate(y))
        # derivatives caused by biological reactions
        bio_derivatives = self.community.get_derivatives(expanded_y, self.T, self.progress_tracker)
        dy_dt_bio = np.sum(bio_derivatives, axis=0)
        # derivatives caused by the chemostat
        dy_dt_chemo = self.D_vector * (self.input - expanded_y)
        # derivatives caused by gas-liquid transfers
        glt_rates = self.system_glt.get_rates(expanded_y, self.T, self.progress_tracker)
        glt_matrix = self.system_glt.get_matrix()
        glt_rate_matrix = np.matmul(np.diag(glt_rates), glt_matrix)
        dy_dt_glt = np.sum(glt_rate_matrix, axis=0)
        dy_dt = dy_dt_bio + dy_dt_chemo + dy_dt_glt
        # derivatives
        return np.vstack([bio_derivatives, dy_dt_chemo, dy_dt_glt])

    # Replacement for the solve function
    # In this new version, integration is performed once (instead of being done by chunks)
    # then the results are interpolated on the specified timepoints
    def solve(self, timepoints, mode="ivp", method="RK45"):
        self.endpoint = max(timepoints)
        self.progress_tracker.set_total_time(self.endpoint)
        expanded_y = self.equilibrate(self.y0)
        assert all(y >= 0 for y in expanded_y)
        self.logger.do_log(self, 0, self.y0, expanded_y)
        self.status = simulation_status["is running"]
        y0 = np.asarray(self.y0)
        if mode == "ivp":
            ivp_ret = solve_ivp(self.f, (0, self.endpoint), y0, t_eval=np.asarray(sorted(timepoints)), method=method, atol=self.atol)
            print()
            print("solve_ivp message: {}".format(ivp_ret.message))
            ys = ivp_ret.y.transpose()
        elif mode == "odeint":
            ys, opt_out = odeint(self.f_for_odeint, self.y0, timepoints, full_output=True, atol=self.atol)
        elif mode == "by-chunk":
            solver = ode(self.f)
            solver.set_integrator("lsoda", atol=self.atol, nsteps=50000)
            solver.set_initial_value(self.y0)
            self.progress_tracker.set_total_time(self.endpoint)
            ts = [0]
            ys = [expanded_y]
            self.logger.do_log(self, 0, self.y0, expanded_y)
            timestep_gen = (step for step in np.diff(timepoints))

            self.status = simulation_status["is running"]
            while solver.successful() and solver.t < self.endpoint:
                solver.integrate(solver.t + next(timestep_gen))
                ts.append(solver.t)
                ys.append(solver.y)
                expanded_y = self.equilibrate(solver.y)
                self.logger.do_log(self, solver.t, solver.y, expanded_y)
                self.progress_tracker.update_progress(solver.t)
            print()
        else:
            raise ValueError("Undefined integration mode \"{}\"".format(mode))
        if self.t >= self.endpoint:
            self.status = simulation_status["has run until the end"]
        else:
            self.status = simulation_status["has run but stopped before the end"]
            print(opt_out)
        print("integration done")

        time_array = np.transpose(timepoints[:len(ys)])
        print("equilibrating the results...")
        conc_array = np.vstack([self.equilibrate(y) for y in ys])
        print("equilibration done")
        data = np.column_stack((time_array, conc_array))

        labels = ["time"] + self.chems_list
        return pd.DataFrame(data=data, columns=labels)

    def diagnose_integration_failure(self, solver):
        if solver.stiff:
            print("The system is stiff")
        else:
            print("The system is not stiff")
        print("Here are the derivatives of the different processes for each variable, sorted by absolute total derivative:")
        header = "{:<14}{}\t{}\t{}".format("derivatives:", "\t".join(population.population_name for population in self.community.populations), "dilution", "gas/liquid transfers")
        diagnosis_derivatives = self.f_diagnosis(solver.t, solver.y).transpose()
        expanded_y = self.equilibrate(solver.y)
        rows = sorted(zip(self.chems_list, diagnosis_derivatives),
                      reverse=True,
                      key=lambda row: abs(sum(row[1])))
        formatted_rows = ("{:<14}{}".format(chem_name, "\t".join("{:.2e}".format(value) for value in values)) for chem_name, values in rows)
        print(header)
        print("\n".join(formatted_rows))
        print("")
        print("Here are the value/derivative ratio of consummed species:")
        consummed_species_rows = sorted(((chem_name, y, sum(dy), y/sum(dy))
                                         for chem_name, y, dy
                                         in zip(self.chems_list, expanded_y, diagnosis_derivatives)
                                         if sum(dy) < 0),
                                         key=lambda row: abs(row[3]))
        print("{:<14}{:<14}{:<14}{:<14}".format("species", "value", "derivative", "ratio"))
        print("\n".join("{:<14}{:<14.2e}{:<14.2e}{:<14.2e}".format(name, y, dy, ratio)
              for name, y, dy, ratio
              in consummed_species_rows))
