from dataclasses import dataclass
from random import randint

from dp.top_down import mwis_top_down
from util.types import Board
from ga import (
    genetic_algorithm,
    crossover,
    mutation,
    q,
    reproduction,
    succession,
)


@dataclass
class Range:
    low: int
    high: int


def generate_board(rows: int, columns: int, r: Range) -> Board:
    return [[randint(r.low, r.high) for _ in range(columns)] for _ in range(rows)]


N_COLUMNS = 4
N_ROWS = 200
LIMITS = Range(-10, 10)
N_CARDS = 20


def main() -> None:
    print("DP")
    board = generate_board(N_ROWS, N_COLUMNS, LIMITS)
    result = mwis_top_down(board, N_CARDS)
    print(f"Board: {board}")
    print(f"Total sum: {result}")
    print("GA")
    result = genetic_algorithm.GeneticAlgorithm(
        q.q,
        mutation.mutation,
        reproduction.reproduction,
        crossover.crossover,
        succession.elitism,
        population_count=150,
        probability_of_crossover=0.8,
        probability_of_mutation=0.04,
        fes=200000,
        num_of_cards=N_CARDS,
        board=board,
        num_of_best_survivors=2,
    ).run()
    print(result[1])


if __name__ == "__main__":
    main()
