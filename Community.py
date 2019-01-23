import numpy as np

class Community:
    def __init__(self, populations):
        self.populations = populations

    def get_derivatives(self, C, T, tracker):
        return np.vstack(population.get_derivatives(C, T, tracker) for population in self.populations)

    def get_population_specific_variables_indexes(self):
        '''Asks all populations for their specific variables' indexes.
        Returns it as a list'''
        indexes = list()
        for population in self.populations:
            for chem_info in population.specific_chems.values():
                indexes.append(population.chems_list.index(chem_info["name"]))
        return indexes

    def get_index_of_chems_unaffected_by_dilution(self):
        indexes = list()
        for population in self.populations:
                    indexes.extend(population.get_index_of_specific_chems_unaffected_by_dilution())
        return indexes

    def pretty_str(self):
        return "\n".join(population.pretty_str() for population in self.populations)

if __name__== "__main__":
    pass
