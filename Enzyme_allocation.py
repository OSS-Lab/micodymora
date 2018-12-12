import numpy as np

def get_naive_enzyme_allocation(params):
    def naive_enzyme_allocation(rates):
        power_rates = np.power(rates, params["n"])
        return power_rates / np.sum(power_rates)
    return naive_enzyme_allocation

enzyme_allocations_dict = {"naive": get_naive_enzyme_allocation}
