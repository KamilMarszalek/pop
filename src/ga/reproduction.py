from random import choice
from typing import Callable

from src.ga.type_definitions import Population
from src.ga.unit import Unit

type ReproducitionFunc = Callable[[Population, dict["Unit", int], int], Population]


def reproduction(
    population: Population, evaluations: dict["Unit", int], population_count: int
) -> Population:
    """tournament"""
    new_population: Population = []
    for _ in range(population_count):
        unit1 = choice(population)
        unit2 = choice(population)
        eval1 = evaluations[unit1]
        eval2 = evaluations[unit2]
        if eval1 >= eval2:
            new_population.append(unit1)
        else:
            new_population.append(unit2)
    return new_population
