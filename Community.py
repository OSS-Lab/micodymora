import numpy as np

class Population:
    '''Description of a population with simulation-specific informations.
    This class is then able to pass messages about the system's state
    (concentration vectors etc) to its catabolisms'''
    def __init__(self, name, catabolisms, enzyme_allocation, anabolism, biomass_index):
        '''
        * name: name of the population
        * catabolisms: list of MetabolicReaction instances representing the catabolisms
            catalyzed by the population
        * enzyme_allocation: function taking specific rate vector and returning 
            a vector of the proportion of biomass allocated to each catabolism
        * anabolism: dictionary containing the parameters of the
        anabolism
        '''
        self.name = name
        self.catabolisms = catabolisms
        self.enzyme_allocation = enzyme_allocation
        self.anabolism = anabolism
        self.biomass_index = biomass_index
        
        # make the metabolic reactions aware of itself
        for catabolism in self.catabolisms:
            catabolism.population = self
        self.anabolism.population = self

    def add_catabolism(self, catabolism):
        '''add a catabolic reaction to the list of the catabolisms the
        population is able to catalyze.
        * catabolism: Reaction.MetabolicReaction instance'''
        self.catabolisms.append(catabolism)

    def get_catabolism_matrix(self, C, T):
        return np.vstack(np.vstack(catabolism.get_vector(C, T) for catabolism in self.catabolisms))

    def get_catabolism_rates(self, C, T):
        '''returns the vector of catabolisms rates at population scale'''
        specific_catabolic_rates = [catabolism.get_rate(C, T) for catabolism in self.catabolisms]
        biomass = C[self.biomass_index]
        return biomass * self.enzyme_allocation(specific_catabolic_rates) * specific_catabolic_rates

    def get_anabolism_rate(self, C, T):
        return self.anabolism.get_rate(C, T)

    def get_matrix(self, C, T):
        catabolism_matrix = self.get_catabolism_matrix(C, T)
        anabolism_vector = self.anabolism.get_vector(C, T)
        return np.vstack([catabolism_matrix, anabolism_vector])

    def get_rates(self, C, T):
        catabolism_rates = self.get_catabolism_rates(C, T)
        anabolism_rate = self.get_anabolism_rate(C, T)
        return np.hstack([catabolism_rates, anabolism_rate])

    def get_population_density(self, C):
        return C[self.biomass_index]

    def __str__(self):
        return self.name

    def pretty_str(self):
        return "{}\n{}".format(self.name, "\n".join("\t" + str(cat.reaction) for cat in self.catabolisms))

class Community:
    def __init__(self, populations):
        self.populations = populations

    def get_matrix(self, C, T):
        return np.vstack(population.get_matrix(C, T) for population in self.populations) 
            
    def get_rates(self, C, T):
        return np.hstack(population.get_rates(C, T) for population in self.populations) 

    def pretty_str(self):
        return "\n".join(population.pretty_str() for population in self.populations)

if __name__== "__main__":
    pass
