from random import choice


class Unit:
    def __init__(self, num_of_rows: int, num_of_cards: int) -> None:
        self.choices = generate_non_adjacent_masks(4)
        self.genes = [choice(self.choices) for _ in range(num_of_rows)]
        self.num_of_cards = num_of_cards

    def repair(self, num_of_cards: int) -> None:
        for i in range(1, len(self.genes)):
            self.genes[i] &= self.genes[i - 1]
        num_of_cards_used = sum(x.bit_count() for x in self.genes)
        if num_of_cards_used > num_of_cards:
            pass


def q(individual: list[list[int]], board: list[list[int]]) -> int:
    reward = 0
    for col_ind, col_board in zip(individual, board):
        for ind, value in zip(col_ind, col_board):
            reward += ind * value
    return reward


def generate_non_adjacent_masks(size: int) -> list[int]:
    valid_masks: list[int] = []
    for mask in range(1 << size):
        if mask & (mask << 1):
            continue
        valid_masks.append(mask)
    return valid_masks


def repair_unit(unit: list[int], num_of_cards: int) -> list[int]:
    for i in range(1, len(unit)):
        unit[i] &= unit[i - 1]


def generate_starting_population(
    population_count: int, num_of_rows: int, num_of_cards: int
) -> list[list[int]]:
    return [Unit(num_of_rows, num_of_cards) for _ in range(population_count)]
