from random import choice, uniform
from copy import deepcopy
import heapq
from type_definitions import Board, Population
from unit import Unit


def q(unit: "Unit", board: Board) -> int:
    reward = 0
    for mask, col_board in zip(unit.genes, board):
        for row in range(4):
            if (mask >> row) & 1:
                reward += col_board[row]
    return reward


def generate_starting_population(
    population_count: int, num_of_columns: int, num_of_cards: int
) -> Population:
    return [Unit(num_of_columns, num_of_cards) for _ in range(population_count)]


def get_population_evaluation(
    population: Population, board: Board
) -> dict["Unit", int]:
    evaluation = {}
    for u in population:
        evaluation[u] = q(u, board)
    return evaluation


def find_best_unit(
    population: Population, evaluations: dict["Unit", int]
) -> tuple["Unit", int]:
    best_unit = population[0]
    best_value = evaluations[best_unit]
    for unit in population:
        evaluation = evaluations[unit]
        if evaluation > best_value:
            best_unit = unit
            best_value = evaluation
    return best_unit, best_value


def stop(t: int, t_max: int):
    return t >= t_max


def reproduction(
    population: Population,
    evaluations: dict["Unit", int],
    population_count: int,
) -> Population:
    """tournament"""
    new_population = []
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


def elitism(
    new_population: Population,
    board: Board,
    old_population: Population = None,
    num_of_best_survivors: int = 0,
):
    if num_of_best_survivors <= 0:
        return new_population
    best_survivors = heapq.nlargest(
        num_of_best_survivors, old_population, key=lambda x: q(x, board)
    )
    new_population.sort(key=lambda x: q(x, board))
    return new_population[num_of_best_survivors:] + best_survivors


def genetic_algorithm(
    q: callable,
    population_count: int,
    probability_of_mutation: float,
    probability_of_crossover: float,
    t_max: int,
    num_of_columns: int,
    num_of_cards: int,
    board: Board,
    num_of_best_survivors: int = 0,
    starting_population: Population = None,
) -> tuple["Unit", int]:
    population = (
        starting_population
        if starting_population
        else generate_starting_population(
            population_count,
            num_of_columns,
            num_of_cards,
        )
    )
    t = 0
    evaluations = get_population_evaluation(population, board)
    best_unit, best_value = find_best_unit(population, evaluations)
    while not stop(t, t_max):
        r = reproduction(population, evaluations, population_count)
        c = crossover(r, probability_of_crossover)
        m = mutation(c, probability_of_mutation)
        m_evaluations = get_population_evaluation(m, board)
        best_candidate, best_candidate_evaluation = find_best_unit(
            m,
            m_evaluations,
        )
        if best_candidate_evaluation > best_value:
            best_unit = best_candidate
            best_value = best_candidate_evaluation
        population = (
            m
            if num_of_best_survivors <= 0
            else elitism(m, population, num_of_best_survivors)
        )
        evaluations = get_population_evaluation(population, board)
        t += 1
    return best_unit, best_value
