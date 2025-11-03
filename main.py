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
N_ROWS = 500
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
        5,
        0.2,
        0.5,
        100000,
        N_CARDS,
        board,
    ).run()
    print(result[1])


if __name__ == "__main__":
    main()
