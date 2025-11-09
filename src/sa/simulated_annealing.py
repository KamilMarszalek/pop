import math
import random
from dataclasses import dataclass

from src.sa.board_state import BoardState
from src.sa.greedy import TileHeuristic, greedy_fill, weight_per_neighbors
from src.sa.neighbor_generator import NeighborGenerator
from src.util.types import Board, MWISResult
from src.util.time_measure import measure_time


@dataclass
class SimulatedAnnealingParams:
    n_iter: int = 2_000
    T0: float = 10_000.0
    T0_threshold: float = 1e-5
    cooling: float = 0.99


@measure_time()
def simulated_annealing(
    board: Board,
    max_cards: int,
    params: SimulatedAnnealingParams,
    generator: NeighborGenerator,
    heuristic: TileHeuristic = weight_per_neighbors,
) -> MWISResult:
    best = BoardState(board)
    greedy_fill(best, max_cards, heuristic)
    best_eval = best.evaluate_sum()
    current, current_eval = best, best_eval
    print(f"Initial greedy: {best_eval}")
    t = params.T0
    for i in range(params.n_iter):
        candidate = generator(current, max_cards)
        candidate_eval = candidate.evaluate_sum()
        if candidate_eval > current_eval or random.uniform(0, 1) < math.exp(
            (candidate_eval - current_eval) / t
        ):
            current, current_eval = candidate, candidate_eval
            if current_eval > best_eval:
                best, best_eval = candidate, candidate_eval
        if i % 100 == 0:
            print(f"Iteration: {i}, best_score: {best_eval}")
        t *= params.cooling
    return best_eval, best.convert_state_to_masks()
