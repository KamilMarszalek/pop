from dataclasses import dataclass

from src.sa.board_state import BoardState
from src.sa.greedy_fill import greedy_fill, weight
from src.sa.successor_generator import FixLocalRegions, SuccessorGenerator
from src.util.time_measure import measure_time
from src.util.types import Board, MWISResult


@dataclass
class GreedyLocalRepairParams:
    n_iter: int = 200
    region_percent_size: float = 0.05


@measure_time()
def greedy_and_repair(
    board: Board,
    max_cards: int,
    params: GreedyLocalRepairParams | None = None,
    generator: SuccessorGenerator | None = None,
) -> MWISResult:
    if params is None:
        params = GreedyLocalRepairParams()
    if generator is None:
        generator = FixLocalRegions(int(params.region_percent_size * len(board)))
    state = BoardState(board)
    greedy_fill(state, max_cards, weight)
    eval = state.evaluate_sum()
    print(f"Initial greedy: {eval}")
    for _ in range(params.n_iter):
        state = generator(state, max_cards)
        eval = state.evaluate_sum()
    return eval, state.convert_state_to_masks()
