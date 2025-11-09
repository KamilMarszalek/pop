from src.astar.state import State
from src.util.types import Board
from src.util.util import generate_non_adjacent_masks
from src.astar.heuristic import Heuristic
from heapq import heappush


class SuccessorGenerator:
    def __init__(
        self,
        num_of_cards: int,
        board: Board,
        heuristics: Heuristic,
        visited: dict,
        queue: list["State"],
    ):
        self.num_of_cards = num_of_cards
        self.board = board
        self.masks = generate_non_adjacent_masks(len(self.board[0]))
        self.heuristics = heuristics
        self.visited = visited
        self.queue = queue

    def generate_successors(
        self,
        state: "State",
        current_best: int,
    ) -> list["State"]:
        if state.col_index >= len(self.board):
            return None
        col = self.board[state.col_index]
        for mask in self._get_valid_masks_for_current_state(state):
            delta_profit, cards_used = self._count_delta_profit(col, mask)
            if cards_used + state.cards_used > self.num_of_cards:
                continue
            new_state = self._build_successor_state(
                mask,
                delta_profit,
                cards_used,
                state,
            )
            if not self._is_state_promising(new_state, current_best):
                continue
            self._enqueue_state(new_state)

    def _get_valid_masks_for_current_state(self, state) -> list[int]:
        return [mask for mask in self.masks if not (mask & state.previous_mask)]

    def _build_successor_state(
        self, mask: int, delta_profit: int, cards_used: int, state: "State"
    ) -> "State":
        return State(
            state.g_reward + delta_profit,
            self.heuristics(
                state.col_index + 1,
                mask,
                self.num_of_cards - state.cards_used - cards_used,
            ),
            state.col_index + 1,
            mask,
            state.cards_used + cards_used,
        )

    def _is_state_promising(self, new_state: State, current_best: int) -> bool:
        if current_best > float("-inf") and new_state.f_reward() <= current_best:
            return False
        old = self.visited.get(new_state.key(), -1)
        return new_state.g_reward > old

    def _enqueue_state(self, state: "State") -> None:
        heappush(self.queue, state)

    def _count_delta_profit(self, col: list[int], mask: int) -> tuple[int, int]:
        delta_profit = 0
        cards_used = 0
        for row in range(len(col)):
            if (mask >> row) & 1:
                delta_profit += col[row]
                cards_used += 1
        return delta_profit, cards_used
