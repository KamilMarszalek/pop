from src.util.types import Board
from heapq import heappush, heappop, nlargest
from src.util.util import generate_non_adjacent_masks


class State:
    def __init__(
        self,
        g_cost: int,
        h_cost: int,
        col_index: int,
        previous_mask: int,
        cards_used: int,
    ) -> None:
        self.col_index = col_index
        self.previous_mask = previous_mask
        self.cards_used = cards_used
        self.g_cost = g_cost
        self.h_cost = h_cost

    def f_cost(self) -> int:
        return self.g_cost + self.f_cost


def h_cost(board: Board, col_index: int, cards_left: int) -> int:
    if cards_left <= 0:
        return 0
    flat_board_remaining_cols_only = [
        cell for row in board[col_index:] for cell in row if cell > 0
    ]
    biggest_values = nlargest(cards_left, flat_board_remaining_cols_only)
    return -sum(biggest_values[:cards_left])


def a_star(board: Board, num_of_cards: int) -> int:
    pass
