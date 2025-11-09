from typing import Protocol
from src.util.types import Board
from src.util.util import generate_non_adjacent_masks


class Heuristic(Protocol):
    def __call__(self, col_index: int, mask: int, cards_left: int) -> int: ...


class BlockDPHeuristic:
    def __init__(
        self,
        board: Board,
        num_of_cards: int,
        blocksize: int = 10,
    ) -> None:

        self.board = board
        self.masks = generate_non_adjacent_masks(len(self.board[0]))
        self.num_of_cards = num_of_cards
        self.blocksize = blocksize
        self.h_table = self._precompute_h_reward()

    def __call__(self, col_index: int, mask: int, cards_left: int) -> int:
        return self.h_table[(col_index, mask, cards_left)]

    def _precompute_h_reward(self) -> dict[tuple[int, int, int], int]:
        mask_values = self._precompute_mask_values()
        global_max_sum = self._compute_global_max_sum()
        precomputed: dict[tuple[int, int, int], int] = self._init_precomputed()

        for start_col in range(len(self.board) - 1, -1, -self.blocksize):
            end_col = min(start_col + self.blocksize, len(self.board))
            self._compute_block_dp(
                start_col,
                end_col,
                precomputed,
                mask_values,
            )

            self._propagate_block_bounds(
                start_col, self.blocksize, precomputed, global_max_sum
            )

        return precomputed

    def _precompute_mask_values(self) -> dict[tuple[int, int], int]:
        mask_values = {}
        for i, col in enumerate(self.board):
            for mask in self.masks:
                s = 0
                for row in range(len(self.board[i])):
                    if (mask >> row) & 1 and col[row] > 0:
                        s += col[row]
                mask_values[(i, mask)] = s
        return mask_values

    def _compute_global_max_sum(self) -> list[int]:
        flat_positive = [v for col in self.board for v in col if v > 0]
        flat_positive.sort(reverse=True)
        global_max_sum = [0] * (self.num_of_cards + 1)
        for k in range(1, self.num_of_cards + 1):
            global_max_sum[k] = sum(flat_positive[:k])
        return global_max_sum

    def _init_precomputed(self) -> dict[tuple[int, int, int], int]:
        precomputed: dict[tuple[int, int, int], int] = {}
        for prev_mask, cards_left in self._iterate_masks_and_cards():
            precomputed[(len(self.board), prev_mask, cards_left)] = 0
        return precomputed

    def _iterate_masks_and_cards(self):
        for prev_mask in self.masks:
            for cards_left in range(self.num_of_cards + 1):
                yield prev_mask, cards_left

    def _compute_block_dp(
        self,
        start_col: int,
        end_col: int,
        precomputed: dict[tuple[int, int, int], int],
        mask_values: dict[tuple[int, int], int],
    ) -> None:
        for i in range(end_col - 1, start_col - 1, -1):
            for prev_mask, cards_left in self._iterate_masks_and_cards():
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
        precomputed: dict[tuple[int, int, int], int],
        global_max_sum: list[int],
    ):
        if start_col > 0:
            prev_block_start = max(0, start_col - block_size)
            for i in range(start_col - 1, prev_block_start - 1, -1):
                for prev_mask, cards_left in self._iterate_masks_and_cards():
                    precomputed[(i, prev_mask, cards_left)] = max(
                        precomputed.get(
                            (i + 1, prev_mask, cards_left),
                            0,
                        ),
                        global_max_sum[cards_left],
                    )
