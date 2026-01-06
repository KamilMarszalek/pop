from random import choice, uniform
from typing import Callable

from src.ga.type_definitions import Population

type CrossoverFunc = Callable[[Population, float, int], Population]


def crossover(
    population: Population, probability_of_crossover: float, population_count: int
) -> Population:
    new_population: Population = []
    while len(new_population) < population_count:
        unit1 = choice(population)
        unit2 = choice(population)
        if uniform(0, 1) < probability_of_crossover:
            new_unit1, new_unit2 = unit1.cross(unit2)
            new_population.extend([new_unit1, new_unit2])
        else:
            new_population.extend([unit1, unit2])

    return new_population[:population_count]
