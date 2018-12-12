'''
This module provides functions helping to convert the aggregated and expanded
vectors in the simulation.

Indeed, during a simulation, two different formats of concentration vectors are
manipulated by the functions defined in this program:

* expanded vectors: containing the concentration of all involved chemical
species in all their different states. For example, the following list
corresponds to an expanded vector: [H+, HO-, Ac, Ac(-)]

* aggregated vectors: containing only the total concentration of each specie in
equilibrium. For example, the aggregated vector corresponding to the expanded
vector of the previous example would be [H+, HO-, Ac], where "Ac" stands for
the total concentration of Acetate (Ac + Ac(-)).
'''

def nest(expanded, nesting):
    return [expanded[nesting[i]:nesting[i+1]] for i in range(len(nesting) - 1)]

def aggregate(expanded, nesting):
    return [sum(n) for n in nest(expanded, nesting)]

if __name__ == "__main__":
    expanded = list(range(6))
    nesting = [0, 1, 3, 5, 6] # (.) (..) (..) (.)
    print("expanded: {}".format(expanded))
    print("with nesting {}:".format(nesting))
    print("nested: {}".format(nest(expanded, nesting)))
    print("aggregated: {}".format(aggregate(expanded, nesting)))
    nesting = [0, 2, 4, 6] # (..) (..) (..)
    print("with nesting {}:".format(nesting))
    print("nested: {}".format(nest(expanded, nesting)))
    print("aggregated: {}".format(aggregate(expanded, nesting)))
