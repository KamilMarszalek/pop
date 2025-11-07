from src.util.types import Board
from heapq import heappush, heappop, nlargest
from src.util.util import generate_non_adjacent_masks
from src.astar.state import State


class AStar:
    def __init__(self, board: Board, num_of_cards: int) -> None:
        self.masks = generate_non_adjacent_masks(4)
        self.board = board
        self.num_of_cards = num_of_cards
        self.precomputed_h_rewards = self._precompute_h_reward()
        self.current_state = self._get_initial_state()
        self.queue = [self.current_state]
        self.visited = {}
        self.best_profit = 0

    def _get_initial_state(self) -> "State":
        return State(0, self.h_reward(0, self.num_of_cards), 0, 0, 0)

    def _precompute_h_reward(self) -> dict[tuple[int, int], int]:
        precomputed = {}
        for col_index in range(len(self.board) + 1):
            for cards in range(self.num_of_cards + 1):
                if cards == 0 or col_index == len(self.board):
                    precomputed[(col_index, cards)] = 0
                    continue
                sliced_flat_board = [
                    cell for row in self.board[col_index:] for cell in row if cell > 0
                ]
                biggest_values = nlargest(cards, sliced_flat_board)
                precomputed[(col_index, cards)] = sum(biggest_values[:cards])
        return precomputed

    def h_reward(self, col_index: int, cards_left: int) -> int:
        return self.precomputed_h_rewards[(col_index, cards_left)]

    def should_continue(self) -> bool:
        return (
            self.current_state.col_index < len(self.board)
            and self.current_state.cards_used <= self.num_of_cards
            and self.queue
        )

    def generate_children(self) -> None:
        if self.current_state.col_index >= len(self.board):
            return None
        col = self.board[self.current_state.col_index]
        for mask in self.masks:
            if mask & self.current_state.previous_mask:
                continue
            delta_profit, cards_used = self._count_delta_profit(col, mask)
            if cards_used + self.current_state.cards_used > self.num_of_cards:
                continue
            new_state = State(
                self.current_state.g_reward + delta_profit,
                self.h_reward(
                    self.current_state.col_index + 1,
                    self.num_of_cards - self.current_state.cards_used - cards_used,
                ),
                self.current_state.col_index + 1,
                mask,
                self.current_state.cards_used + cards_used,
            )
            old = self.visited.get(
                (new_state.col_index, new_state.previous_mask, new_state.cards_used), -1
            )
            if old >= new_state.g_reward:
                continue

            heappush(self.queue, new_state)

    def _count_delta_profit(self, col: list[int], mask: int) -> tuple[int, int]:
        delta_profit = 0
        cards_used = 0
        for row in range(4):
            if (mask >> row) & 1:
                delta_profit += col[row]
                cards_used += 1
        return delta_profit, cards_used

    def run(self) -> int:
        while self.should_continue():
            self.current_state = heappop(self.queue)
            self.visited[
                (
                    self.current_state.col_index,
                    self.current_state.previous_mask,
                    self.current_state.cards_used,
                )
            ] = self.current_state.g_reward
            self.generate_children()

            if self.current_state.g_reward > self.best_profit:
                self.best_profit = self.current_state.g_reward
        return max(self.best_profit, self.current_state.g_reward)
