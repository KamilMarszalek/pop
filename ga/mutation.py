from type_definitions import Population
from copy import deepcopy


def mutation(
    population: Population,
    probability_of_mutation: float,
) -> Population:
    population_copy = deepcopy(population)
    new_population = []
    for unit in population_copy:
        unit.mutate(probability_of_mutation)
        new_population.append(unit)
    return new_population
