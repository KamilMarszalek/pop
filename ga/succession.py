from .type_definitions import Population
from .unit import Unit
import heapq


def elitism(
    m_population: Population,
    m_evaluation: dict["Unit", int],
    old_population: Population = None,
    old_evaluation: dict["Unit", int] = None,
    num_of_best_survivors: int = 0,
):
    if num_of_best_survivors <= 0:
        return m_population, m_evaluation
    best_survivors = heapq.nlargest(
        num_of_best_survivors, old_population, key=lambda x: old_evaluation[x]
    )
    m_population.sort(key=lambda x: m_evaluation[x])
    new_population = m_population[num_of_best_survivors:] + best_survivors
    new_evaluation = {}
    for unit in new_population:
        if unit in m_evaluation:
            new_evaluation[unit] = m_evaluation[unit]
        elif unit in old_evaluation:
            new_evaluation[unit] = old_evaluation[unit]
    return new_population, new_evaluation
