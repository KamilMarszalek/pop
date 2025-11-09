from src.util.types import Board
from heapq import heappush, heappop
from src.util.util import generate_non_adjacent_masks
from src.astar.state import State
from src.util.time_measure import measure_time


@measure_time()
class AStar:
    def __init__(self, board: Board, num_of_cards: int) -> None:
        self.masks = generate_non_adjacent_masks(4)
        self.board = board
        self.num_of_cards = num_of_cards
        self.precomputed_h_rewards = self._precompute_h_reward_block_dp()
        self.current_state = self._get_initial_state()
        self.queue = [self.current_state]
        self.visited = {}
        self.best_profit = float("-inf")

    def _get_initial_state(self) -> "State":
        return State(0, self.h_reward(0, 0, self.num_of_cards), 0, 0, 0)

    def _precompute_h_reward_full_dp(self) -> dict[tuple[int, int, int], int]:
        mask_values: dict[tuple[int, int], int] = self._precompute_mask_values()
        precomputed: dict[tuple[int, int, int], int] = {}
        for i in range(len(self.board), -1, -1):
            for prev_mask in self.masks:
                for cards_left in range(self.num_of_cards + 1):
                    if i == len(self.board) or cards_left == 0:
                        precomputed[(i, prev_mask, cards_left)] = 0
                    else:
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
                            best = max(best, value)
                        precomputed[(i, prev_mask, cards_left)] = best
        print(precomputed[(0, 0, self.num_of_cards)])
        return precomputed

    def _compute_global_max_sum(self) -> list[int]:
        flat_positive = [v for col in self.board for v in col if v > 0]
        flat_positive.sort(reverse=True)
        global_max_sum = [0] * (self.num_of_cards + 1)
        for k in range(1, self.num_of_cards + 1):
            global_max_sum[k] = sum(flat_positive[:k])
        return global_max_sum

    def _precompute_h_reward_lookahead(
        self, depth: int = 5
    ) -> dict[tuple[int, int, int], int]:
        self.mask_values = self._precompute_mask_values()
        precomputed = {}
        self.memo = {}
        self.global_max_sum = self._compute_global_max_sum()
        for i in range(len(self.board), -1, -1):
            for prev_mask in self.masks:
                for cards_left in range(self.num_of_cards + 1):
                    if i == len(self.board) or cards_left == 0:
                        precomputed[(i, prev_mask, cards_left)] = 0
                        continue

                    precomputed[(i, prev_mask, cards_left)] = self.__dp_local(
                        i,
                        prev_mask,
                        cards_left,
                        depth,
                    )
        return precomputed

    def __dp_local(
        self,
        col_idx: int,
        prev_mask: int,
        cards_left: int,
        depth: int,
    ) -> int:
        if (col_idx, prev_mask, cards_left) in self.memo:
            return self.memo[(col_idx, prev_mask, cards_left)]
        if col_idx == len(self.board) or cards_left == 0 or depth == 0:
            res = self.global_max_sum[min(cards_left, self.num_of_cards)]
            self.memo[(col_idx, prev_mask, cards_left)] = res
            return res
        local_best = 0
        for mask in self.masks:
            if mask & prev_mask:
                continue
            used = mask.bit_count()
            if used > cards_left:
                continue
            value = self.mask_values[(col_idx, mask)] + self.__dp_local(
                col_idx + 1,
                mask,
                cards_left - used,
                depth - 1,
            )
            if value > local_best:
                local_best = value
        self.memo[(col_idx, prev_mask, cards_left)] = local_best
        return local_best

    def _precompute_mask_values(self) -> dict[tuple[int, int], int]:
        mask_values = {}
        for i, col in enumerate(self.board):
            for mask in self.masks:
                s = 0
                for row in range(4):
                    if (mask >> row) & 1 and col[row] > 0:
                        s += col[row]
                mask_values[(i, mask)] = s
        return mask_values

    def _precompute_h_reward_block_dp(
        self, block_size: int = 10
    ) -> dict[tuple[int, int, int], int]:
        mask_values = self._precompute_mask_values()
        global_max_sum = self._compute_global_max_sum()
        precomputed: dict[tuple[int, int, int], int] = {}

        for prev_mask in self.masks:
            for cards_left in range(self.num_of_cards + 1):
                precomputed[(len(self.board), prev_mask, cards_left)] = 0

        for start_col in range(len(self.board) - 1, -1, -block_size):
            end_col = min(start_col + block_size, len(self.board))

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

            if start_col > 0:
                prev_block_start = max(0, start_col - block_size)
                for i in range(start_col - 1, prev_block_start - 1, -1):
                    for prev_mask in self.masks:
                        for cards_left in range(self.num_of_cards + 1):
                            precomputed[(i, prev_mask, cards_left)] = max(
                                precomputed.get(
                                    (i + 1, prev_mask, cards_left),
                                    0,
                                ),
                                global_max_sum[cards_left],
                            )

        return precomputed

    def h_reward(self, col_index: int, mask: int, cards_left: int) -> int:
        return self.precomputed_h_rewards[(col_index, mask, cards_left)]

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
                    mask,
                    self.num_of_cards - self.current_state.cards_used - cards_used,
                ),
                self.current_state.col_index + 1,
                mask,
                self.current_state.cards_used + cards_used,
            )
            if (
                self.best_profit > float("-inf")
                and new_state.f_reward() <= self.best_profit
            ):
                continue
            old = self.visited.get(new_state.key(), -1)
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
        while self.queue:
            self.current_state = heappop(self.queue)
            if (
                self.current_state.col_index == len(self.board)
                or self.current_state.cards_used == self.num_of_cards
            ):
                self.best_profit = max(self.best_profit, self.current_state.g_reward)
                continue
            self.visited[self.current_state.key()] = self.current_state.g_reward
            self.generate_children()
        return self.best_profit
