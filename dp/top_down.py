from util.util import (
    calculate_row_sum,
    generate_non_adjacent_masks,
    get_masks_bit_count,
    get_masks_compatibility,
)

type Tabulation = dict[tuple[int, int], int]


def create_tabulation(masks: list[int]) -> Tabulation:
    return {(m, 0): 0 for m in masks}


def mwis_top_down(board: list[list[int]], max_cards: int) -> int:
    possible_masks = generate_non_adjacent_masks(len(board[0]))
    masks_bit_count = get_masks_bit_count(possible_masks)
    compatibility = get_masks_compatibility(possible_masks)
    tab = create_tabulation(possible_masks)

    for row_index in range(len(board) - 1, -1, -1):
        next_tab: Tabulation = {}
        for (previous_mask, cards_used), value in tab.items():
            for mask in compatibility[previous_mask]:
                c = cards_used + masks_bit_count[mask]
                if c > max_cards:
                    continue
                key = (mask, c)
                new_sum = value + calculate_row_sum(board[row_index], mask)
                if new_sum >= next_tab.get(key, 0):
                    next_tab[key] = new_sum
        tab = next_tab

    return max(tab.values())
