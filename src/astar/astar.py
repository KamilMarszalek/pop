from heapq import heappop, heappush

from src.astar.state import State
from src.util.time_measure import measure_time
from src.util.types import Board, MWISResult
from src.util.util import generate_non_adjacent_masks

type EvaluatedValues = dict[tuple[int, int, int], int]
type MaskValues = dict[tuple[int, int], int]


class AStar:
    def __init__(self, board: Board, num_of_cards: int) -> None:
        self.board = board
        self.num_of_rows = len(board[0]) if board else 0
        self.masks = generate_non_adjacent_masks(self.num_of_rows)
        self.num_of_cards = num_of_cards
        self.precomputed_h_rewards = self._precompute_h_reward_block_dp()
        self.current_state = self._get_initial_state()
        self.queue = [self.current_state]
        self.best_profit = float("-inf")
        self.visited: EvaluatedValues = {}

    def _get_initial_state(self) -> "State":
        return State(0, self.h_reward(0, 0, self.num_of_cards), 0, 0, 0)

    def _compute_global_max_sum(self) -> list[int]:
        flat_positive = [v for col in self.board for v in col if v > 0]
        flat_positive.sort(reverse=True)
        global_max_sum = [0] * (self.num_of_cards + 1)
        for k in range(1, self.num_of_cards + 1):
            global_max_sum[k] = sum(flat_positive[:k])
        return global_max_sum

    def _precompute_mask_values(self) -> MaskValues:
        mask_values: MaskValues = {}
        for i, col in enumerate(self.board):
            for mask in self.masks:
                s = 0
                for row in range(self.num_of_rows):
                    if (mask >> (len(col) - row - 1)) & 1 and col[row] > 0:
                        s += col[row]
                mask_values[(i, mask)] = s
        return mask_values

    def _precompute_h_reward_block_dp(self, block_size: int = 10) -> EvaluatedValues:
        mask_values = self._precompute_mask_values()
        global_max_sum = self._compute_global_max_sum()
        precomputed: EvaluatedValues = self._init_precomputed()

        for start_col in range(len(self.board) - 1, -1, -block_size):
            end_col = min(start_col + block_size, len(self.board))
            self._compute_block_dp(start_col, end_col, precomputed, mask_values)

            self._propagate_block_bounds(start_col, block_size, global_max_sum, precomputed)

        return precomputed

    def _init_precomputed(self) -> EvaluatedValues:
        precomputed: EvaluatedValues = {}
        for prev_mask in self.masks:
            for cards_left in range(self.num_of_cards + 1):
                precomputed[(len(self.board), prev_mask, cards_left)] = 0
        return precomputed

    def _compute_block_dp(
        self, start_col: int, end_col: int, precomputed: EvaluatedValues, mask_values: MaskValues
    ) -> None:
        for i in range(end_col - 1, start_col - 1, -1):
            for prev_mask in self.masks:
                for cards_left in range(self.num_of_cards + 1):
                    if cards_left == 0:
                        precomputed[(i, prev_mask, cards_left)] = 0
                        continue

                    best = 0
                    for mask in self.masks:
                        if mask & prev_mask:
                            continue
                        used = mask.bit_count()
                        if used > cards_left:
                            continue
                        value = mask_values[(i, mask)] + precomputed.get(
                            (i + 1, mask, cards_left - used), 0
                        )
                        if value > best:
                            best = value
                    precomputed[(i, prev_mask, cards_left)] = best

    def _propagate_block_bounds(
        self,
        start_col: int,
        block_size: int,
        global_max_sum: list[int],
        precomputed: EvaluatedValues,
    ):
        if start_col > 0:
            prev_block_start = max(0, start_col - block_size)
            for i in range(start_col - 1, prev_block_start - 1, -1):
                for prev_mask in self.masks:
                    for cards_left in range(self.num_of_cards + 1):
                        precomputed[(i, prev_mask, cards_left)] = max(
                            precomputed.get((i + 1, prev_mask, cards_left), 0),
                            global_max_sum[cards_left],
                        )

    def h_reward(self, col_index: int, mask: int, cards_left: int) -> int:
        return self.precomputed_h_rewards[(col_index, mask, cards_left)]

    def generate_children(self) -> None:
        if self.current_state.col_index >= len(self.board):
            return None
        col = self.board[self.current_state.col_index]
        for mask in self._get_valid_masks_for_current_state():
            delta_profit, cards_used = self._count_delta_profit(col, mask)
            if cards_used + self.current_state.cards_used > self.num_of_cards:
                continue
            new_state = self._build_successor_state(mask, delta_profit, cards_used)
            if not self._is_state_promising(new_state):
                continue
            self._enqueue_state(new_state)

    def _get_valid_masks_for_current_state(self) -> list[int]:
        return [mask for mask in self.masks if not (mask & self.current_state.previous_mask)]

    def _build_successor_state(self, mask: int, delta_profit: int, cards_used: int) -> "State":
        return State(
            self.current_state.g_reward + delta_profit,
            self.h_reward(
                self.current_state.col_index + 1,
                mask,
                self.num_of_cards - self.current_state.cards_used - cards_used,
            ),
            self.current_state.col_index + 1,
            mask,
            self.current_state.cards_used + cards_used,
            self.current_state,
        )

    def _is_state_promising(self, new_state: State) -> bool:
        if self.best_profit > float("-inf") and new_state.f_reward() <= self.best_profit:
            return False
        old = self.visited.get(new_state.key(), -1)
        return new_state.g_reward > old

    def _enqueue_state(self, state: "State") -> None:
        heappush(self.queue, state)

    def _count_delta_profit(self, col: list[int], mask: int) -> tuple[int, int]:
        delta_profit = 0
        cards_used = 0
        for row in range(self.num_of_rows):
            if (mask >> (len(col) - row - 1)) & 1:
                delta_profit += col[row]
                cards_used += 1
        return delta_profit, cards_used

    def _reconstruct_path(self, best_state: "State") -> list[int]:
        path: list[int] = []
        s = best_state

        while s is not None and s.col_index > 0:
            path.append(s.previous_mask)
            s = s.parent

        path.reverse()
        while len(path) < len(self.board):
            path.append(0)
        return path

    def run(self) -> MWISResult:
        best_state = None
        while self.queue:
            self.current_state = heappop(self.queue)
            if (
                self.current_state.col_index == len(self.board)
                or self.current_state.cards_used == self.num_of_cards
            ):
                if self.current_state.g_reward > self.best_profit:
                    self.best_profit = self.current_state.g_reward
                    best_state = self.current_state
                continue
            self.visited[self.current_state.key()] = self.current_state.g_reward
            self.generate_children()
        assert best_state is not None
        path = self._reconstruct_path(best_state)
        return int(self.best_profit), path


@measure_time()
def run_astar(board: Board, max_cards: int) -> MWISResult:
    astar = AStar(board, max_cards)
    return astar.run()
