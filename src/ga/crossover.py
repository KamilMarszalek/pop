from src.ga.type_definitions import Population
from random import choice, uniform


def crossover(
    population: Population,
    probability_of_crossover: float,
    population_count: int,
) -> Population:
    new_population = []
    while len(new_population) < population_count:
        unit1 = choice(population)
        unit2 = choice(population)
        if uniform(0, 1) < probability_of_crossover:
            new_unit1, new_unit2 = unit1.cross(unit2)
            new_population.extend([new_unit1, new_unit2])
        else:
            new_population.extend([unit1, unit2])

    return new_population[:population_count]
