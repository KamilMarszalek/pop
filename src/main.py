from dataclasses import dataclass
from random import randint

from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.sa.neighbor_generator import FixLocalRegions
from src.sa.simulated_annealing import SimulatedAnnealingParams, simulated_annealing
from src.util.types import Board


@dataclass
class Range:
    low: int
    high: int


def generate_board(rows: int, columns: int, r: Range) -> Board:
    return [[randint(r.low, r.high) for _ in range(columns)] for _ in range(rows)]


N_COLUMNS = 4
N_ROWS = 800
LIMITS = Range(-10000, 10000)
N_CARDS = 1000


def main() -> None:
    board = [[10, 9], [9, -10]]
    print("---------------------")
    print("DP Top down")
    result, path = mwis_top_down(board, N_CARDS)
    print(f"Result: {result}")
    # for i in path:
    #     print(f"{i:04b}")
    print("---------------------")
    print("DP bottom up")
    result, path = mwis_bottom_up(board, N_CARDS)
    print(f"Result: {result}")
    # for i in path:
    #     print(f"{i:04b}")
    print("---------------------")
    print("Simulated Annealing")
    result, path = simulated_annealing(
        board, N_CARDS, SimulatedAnnealingParams(), FixLocalRegions(int(0.05 * N_ROWS))
    )
    print(f"Total sum: {result}\n")

    print("---------------------")
    print("Genetic algorithm")
    result = genetic_algorithm.GeneticAlgorithm(
        q.q,
        mutation.mutation,
        reproduction.reproduction,
        crossover.crossover,
        succession.elitism,
        population_count=150,
        probability_of_crossover=0.99,
        probability_of_mutation=0.01,
        fes=2_000_000_000,
        num_of_cards=N_CARDS,
        board=board,
        num_of_best_survivors=2,
    ).run()
    print("Result:", result[1])

    print("---------------------")
    print("A*")
    result = AStar(board, N_CARDS).run()
    print("Result:", result)


if __name__ == "__main__":
    main()
