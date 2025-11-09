from src.util.types import Board, MWISResult
from src.util.util import (
    calculate_row_sum,
    generate_non_adjacent_masks,
    get_masks_bit_count,
    get_masks_compatibility,
    merge_compatibility,
)
from src.util.time_measure import measure_time

type Tabulation = dict[tuple[int, int], tuple[int, list[int]]]


def create_tabulation(masks: list[int]) -> Tabulation:
    return {(m, 0): (0, []) for m in masks}


@measure_time()
def mwis_bottom_up(
    board: Board,
    max_cards: int,
    initial_mask: int = 0,
    final_mask: int = 0,
) -> MWISResult:
    possible_masks = generate_non_adjacent_masks(len(board[0]))
    masks_bit_count = get_masks_bit_count(possible_masks)
    compatibility = get_masks_compatibility(possible_masks)
    tab = create_tabulation(possible_masks)

    for row_index in range(len(board) - 1, -1, -1):
        next_tab: Tabulation = {}
        for (previous_mask, cards_used), (accumulated_sum, path) in tab.items():
            comp = compatibility[previous_mask]
            if row_index == 0:
                comp = merge_compatibility(comp, compatibility[initial_mask])
            elif row_index == len(board) - 1:
                comp = merge_compatibility(comp, compatibility[final_mask])
            for mask in comp:
                c = cards_used + masks_bit_count[mask]
                if c > max_cards:
                    continue
                key = (mask, c)
                new_sum = accumulated_sum + calculate_row_sum(board[row_index], mask)
                max_sum, _ = next_tab.get(key, (0, []))
                if new_sum >= max_sum:
                    next_tab[key] = new_sum, [mask] + path
        tab = next_tab
    return max(tab.values(), key=lambda x: x[0])
