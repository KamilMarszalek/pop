import math
import random
from dataclasses import dataclass

from sa.board_state import BoardState
from sa.neighbor_selection_func import NeighborSelectionFunc, remove_and_select
from util.types import Board


@dataclass
class SimulatedAnnealingParams:
    n_iter: int = 1_000
    T0: float = 10.0
    T0_threshold: float = 1e-3
    cooling: float = 0.99


def simulated_annealing(
    board: Board,
    max_cards: int,
    params: SimulatedAnnealingParams,
    neighbor: NeighborSelectionFunc = remove_and_select,
) -> int:
    x = BoardState(board)
    x.greedy_fill(max_cards)
    x_eval = x.evaluate_sum()
    t = params.T0
    for _ in range(params.n_iter):
        y = neighbor(x)
        y_eval = y.evaluate_sum()
        if y_eval > x_eval:
            x, x_eval = y, y_eval
        elif random.uniform(0, 1) < math.exp(-abs(y_eval - x_eval) / t):
            x, x_eval = y, y_eval
        t *= params.cooling
        if t < params.T0_threshold:
            break
    return x_eval
