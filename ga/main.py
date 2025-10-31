from random import choice, uniform
from itertools import combinations


class Unit:
    def __init__(
        self,
        num_of_rows: int,
        num_of_cards: int,
        genes: list[list[int]] = None,
    ) -> None:
        self.choices = generate_non_adjacent_masks(4)
        self.genes = (
            [choice(self.choices) for _ in range(num_of_rows)] if not genes else genes
        )
        self.num_of_cards = num_of_cards
        self.repair()

    def repair(self) -> None:
        for i in range(1, len(self.genes)):
            self.genes[i] &= ~self.genes[i - 1]
        num_of_cards_used = sum(x.bit_count() for x in self.genes)
        while num_of_cards_used > self.num_of_cards:
            column = choice([i for i, g in enumerate(self.genes) if g != 0])
            ones = [bit for bit in range(4) if (self.genes[column] >> bit) & 1]
            bit_to_remove = choice(ones)
            self.genes[column] &= ~(1 << bit_to_remove)
            num_of_cards_used -= 1

    def __str__(self) -> str:
        return str(self.genes)

    def __repr__(self) -> str:
        return str(self.genes)

    def cross(self, other: "Unit") -> tuple["Unit", "Unit"]:
        num_of_crossing_points = 2 if len(self.genes) > 2 else 1
        num_of_rows = len(self.genes)
        crossing_points = choice(
            list(
                combinations(
                    [x for x in range(1, num_of_rows - 1)], num_of_crossing_points
                )
            )
        )
        if num_of_crossing_points == 1:
            new_unit1 = Unit(
                num_of_rows,
                self.num_of_cards,
                self.genes[: crossing_points[0]] + other.genes[crossing_points[0] :],
            )
            new_unit2 = Unit(
                num_of_rows,
                self.num_of_cards,
                other.genes[: crossing_points[0]] + self.genes[crossing_points[0] :],
            )
            return new_unit1, new_unit2
        elif num_of_crossing_points == 2:
            new_unit1 = Unit(
                num_of_rows,
                self.num_of_cards,
                self.genes[: crossing_points[0]]
                + other.genes[crossing_points[0] : crossing_points[1]]
                + self.genes[crossing_points[1] :],
            )
            new_unit2 = Unit(
                num_of_rows,
                self.num_of_cards,
                other.genes[: crossing_points[0]]
                + self.genes[crossing_points[0] : crossing_points[1]]
                + other.genes[crossing_points[1] :],
            )


def q(unit: "Unit", board: list[list[int]]) -> int:
    reward = 0
    for mask, col_board in zip(unit, board):
        bits = [int(bit) for bit in bin(mask)]
        for bit, value in zip(bits, col_board):
            reward += bit * value
    return reward


def generate_non_adjacent_masks(size: int) -> list[int]:
    valid_masks: list[int] = []
    for mask in range(1 << size):
        if mask & (mask << 1):
            continue
        valid_masks.append(mask)
    return valid_masks


def generate_starting_population(
    population_count: int, num_of_rows: int, num_of_cards: int
) -> list["Unit"]:
    return [Unit(num_of_rows, num_of_cards) for _ in range(population_count)]


def get_population_evaluation(
    population: list["Unit"], board: list[list[int]]
) -> dict["Unit", int]:
    evaluation = {}
    for u in population:
        evaluation[u] = q(u, board)
    return evaluation


def find_best_unit(
    population: list["Unit"], evaluations: dict["Unit", int]
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
    return t > t_max


def reproduction(
    population: list["Unit"],
    evaluations: dict["Unit", int],
    population_count: int,
) -> list["Unit"]:
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


def crossing(
    population: list["Unit"], probability_of_crossing: float, population_count: int
) -> list["Unit"]:
    new_population = []
    while len(new_population) < population_count:
        unit1 = choice(population)
        population.remove(unit1)
        if uniform(0, 1) < probability_of_crossing:
            new_unit1, new_unit2 = unit1.cross(unit2)


def genetic_algorithm(
    q: callable,
    population_count: int,
    probability_of_mutation: float,
    probability_of_crossing: float,
    t_max: int,
    num_of_rows: int,
    num_of_cards: int,
    starting_population: list["Unit"] = None,
) -> list[list[int]]:
    population = (
        starting_population
        if starting_population
        else generate_starting_population(
            population_count,
            num_of_rows,
            num_of_cards,
        )
    )
    t = 0
    evaluations = get_population_evaluation(population)
    best_unit, best_value = find_best_unit(population, evaluations)
    while not stop(t, t_max):
        r = reproduction(population, evaluations, population_count)
        c = crossing(r, probability_of_crossing)
