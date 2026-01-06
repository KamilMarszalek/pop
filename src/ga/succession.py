import heapq
from typing import Callable

from src.ga.type_definitions import Population
from src.ga.unit import Unit

type SuccesionFuc = Callable[
    [Population, dict[Unit, int], Population, dict[Unit, int], int],
    tuple[Population, dict[Unit, int]],
]


def elitism(
    m_population: Population,
    m_evaluation: dict[Unit, int],
    old_population: Population,
    old_evaluation: dict[Unit, int],
    num_of_best_survivors: int,
) -> tuple[Population, dict[Unit, int]]:
    if num_of_best_survivors <= 0:
        return m_population, m_evaluation
    best_survivors = heapq.nlargest(
        num_of_best_survivors, old_population, key=lambda x: old_evaluation[x]
    )
    m_population.sort(key=lambda x: m_evaluation[x])
    new_population = m_population[num_of_best_survivors:] + best_survivors
    new_evaluation: dict["Unit", int] = {}
    for unit in new_population:
        if unit in m_evaluation:
            new_evaluation[unit] = m_evaluation[unit]
        elif unit in old_evaluation:
            new_evaluation[unit] = old_evaluation[unit]
    return new_population, new_evaluation
