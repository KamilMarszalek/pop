import math
import random
from dataclasses import dataclass

from sa.board_state import BoardState
from sa.neighbor_selection_func import NeighborSelectionFunc, remove_and_select
from util.types import Board


@dataclass
class SimulatedAnnealingParams:
    n_iter: int = 10_000
    T0: float = 1000.0
    T0_threshold: float = 1e-5
    cooling: float = 0.95


def simulated_annealing(
    board: Board,
    max_cards: int,
    params: SimulatedAnnealingParams,
    neighbor: NeighborSelectionFunc = remove_and_select,
) -> int:
    best = BoardState(board)
    best.greedy_fill(max_cards)
    best_eval = best.evaluate_sum()
    current, current_eval = best, best_eval
    print(f"Initial greedy: {best_eval}")
    t = params.T0
    for _ in range(params.n_iter):
        candidate = neighbor(current, max_cards)
        candidate_eval = candidate.evaluate_sum()
        if candidate_eval > current_eval or random.uniform(0, 1) < math.exp(
            -abs(candidate_eval - best_eval) / t
        ):
            current, current_eval = candidate, candidate_eval
            if current_eval > best_eval:
                best_eval = current_eval
        t *= params.cooling
        if t < params.T0_threshold:
            break
    return best_eval
