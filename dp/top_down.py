from util.types import Board
from util.util import (
    calculate_row_sum,
    generate_non_adjacent_masks,
    get_masks_bit_count,
    get_masks_compatibility,
    merge_compatibility,
)

type Memoization = dict[int, dict[int, int]]


def create_memoization(n_rows: int, possible_masks: list[int]) -> Memoization:
    return {i: {mask: 0 for mask in possible_masks} for i in range(n_rows)}


def mwis_top_down(
    board: Board, max_cards: int, initial_mask: int = 0, final_mask: int = 0
) -> int:
    possible_masks = generate_non_adjacent_masks(len(board[0]))
    compatibility = get_masks_compatibility(possible_masks)
    masks_bit_count = get_masks_bit_count(possible_masks)
    memo = create_memoization(len(board), possible_masks)

    def dsf(row_index: int, cards_used: int, previous_mask: int) -> int:
        if row_index == len(board) or cards_used == max_cards:
            return 0
        if memo[row_index][previous_mask] == 0:
            max_sum = 0
            comp = compatibility[previous_mask]
            if row_index == len(board) - 1:
                comp = merge_compatibility(
                    compatibility[previous_mask], compatibility[final_mask]
                )
            for mask in comp:
                c = cards_used + masks_bit_count[mask]
                if c <= max_cards:
                    max_sum = max(
                        max_sum,
                        calculate_row_sum(board[row_index], mask)
                        + dsf(row_index + 1, c, mask),
                    )
            memo[row_index][previous_mask] = max_sum
        return memo[row_index][previous_mask]

    return int(dsf(0, 0, initial_mask))
