from dataclasses import dataclass

from util.util import (
    calculate_row_sum,
    generate_non_adjacent_masks,
)


@dataclass
class Tabulation:
    current_row: list[list[float]]
    next_row: list[list[float]]


def create_tabulation(n_masks: int, n_cards: int) -> Tabulation:
    current_row = [[float("-inf")] * n_cards for _ in range(n_masks)]
    next_row = [[0.0] * n_cards for _ in range(n_masks)]
    return Tabulation(current_row, next_row)


def mwis_top_down(board: list[list[int]], max_cards: int) -> int:
    possible_masks = generate_non_adjacent_masks(len(board[0]))
    tab = create_tabulation(len(possible_masks), max_cards + 1)

    for row_index in range(len(board) - 1, -1, -1):
        for c in range(max_cards + 1):
            for i, previous_mask in enumerate(possible_masks):
                max_sum = float("-inf")
                for j, mask in enumerate(possible_masks):
                    cards_used = mask.bit_count() + c
                    if not (previous_mask & mask) and cards_used <= max_cards:
                        max_sum = max(
                            max_sum,
                            calculate_row_sum(board[row_index], mask)
                            + tab.next_row[j][cards_used],
                        )
                tab.current_row[i][c] = max_sum
        tab.next_row = tab.current_row
        tab.current_row = [[float("-inf")] * (max_cards + 1) for _ in possible_masks]

    return int(tab.next_row[0][0])
