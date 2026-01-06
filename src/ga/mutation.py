from typing import Callable

from src.ga.type_definitions import Population

type MutationFunc = Callable[[Population, float], Population]


def mutation(population: Population, probability_of_mutation: float) -> Population:
    new_population: Population = []
    for unit in population:
        new_unit = unit.mutate(probability_of_mutation)
        new_population.append(new_unit)
    return new_population
