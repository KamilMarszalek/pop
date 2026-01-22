import random

from src.greedy.board_state import BoardState
from src.greedy.greedy_fill import greedy_fill, weight
from src.greedy.successor_generator import FixLocalRegions
from src.util.time_measure import measure_time
from src.util.types import Board, MWISResult


@measure_time()
def greedy_and_repair(
    board: Board,
    max_cards: int,
    *,
    n_iter: int = 200,
    region_percent_size: float = 0.05,
    rng: random.Random,
) -> MWISResult:
    region_size = max(int(region_percent_size * len(board)), 2)
    generator = FixLocalRegions(region_size, rng)
    state = BoardState(board)
    greedy_fill(state, max_cards, weight)
    eval = state.evaluate_sum()
    evals: list[float] = [eval]
    for _ in range(n_iter):
        state = generator(state, max_cards)
        eval = state.evaluate_sum()
        evals.append(eval)
    return eval, state.convert_state_to_masks(), ([i for i in range(n_iter + 1)], evals)
