from dataclasses import dataclass
from random import randint

from src.astar.astar import AStar
from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.ga import (
    genetic_algorithm,
    crossover,
    mutation,
    succession,
    q,
    reproduction,
)
from src.sa.greedy_and_repair import greedy_and_repair
from src.util.types import Board


@dataclass
class Range:
    low: int
    high: int


def generate_board(rows: int, columns: int, r: Range) -> Board:
    return [[randint(r.low, r.high) for _ in range(columns)] for _ in range(rows)]


N_COLUMNS = 4
N_ROWS = 10
LIMITS = Range(-10000, 10000)
N_CARDS = 5


def main() -> None:
    board = generate_board(N_ROWS, N_COLUMNS, LIMITS)
    print("---------------------")
    print("DP Top down")
    result, path = mwis_top_down(board, N_CARDS)
    print(f"Result: {result}")
    for i in path:
        print(f"{i:04b}")

    print("---------------------")
    print("DP bottom up")
    result, path = mwis_bottom_up(board, N_CARDS)
    print(f"Result: {result}")
    for i in path:
        print(f"{i:04b}")

    print("---------------------")
    print("Greedy with local repair")
    result, path = greedy_and_repair(board, N_CARDS)
    print(f"Total sum: {result}\n")

    print("---------------------")
    print("Genetic algorithm")
    result, path = genetic_algorithm.GeneticAlgorithm(
        q.q,
        mutation.mutation,
        reproduction.reproduction,
        crossover.crossover,
        succession.elitism,
        population_count=150,
        probability_of_crossover=0.99,
        probability_of_mutation=0.01,
        fes=20000,
        num_of_cards=N_CARDS,
        board=board,
        num_of_best_survivors=2,
    ).run()
    print("Result:", result)
    for i in path:
        print(f"{i:04b}")

    print("---------------------")
    print("A*")
    result, path = AStar(board, N_CARDS).run()
    print("Result:", result)
    for i in path:
        print(f"{i:04b}")


if __name__ == "__main__":
    main()
