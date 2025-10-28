from dp.util import calculate_row_sum, generate_non_adjacent_masks

type Memoization = dict[int, dict[int, float]]


def create_memoization(n_rows: int, possible_masks: list[int]) -> Memoization:
    return {i: {mask: float("-inf") for mask in possible_masks} for i in range(n_rows)}


def mwis_bottom_up(board: list[list[int]], max_cards: int) -> int:
    possible_masks = generate_non_adjacent_masks(len(board[0]))
    memo = create_memoization(len(board), possible_masks)

    def dsf(row_index: int, previous_mask: int, cards_used: int) -> float:
        if row_index == len(board) or cards_used == max_cards:
            return 0
        if memo[row_index][previous_mask] == float("-inf"):
            max_sum = float("-inf")
            for mask in possible_masks:
                bits = mask.bit_count()
                if not (previous_mask & mask) and bits + cards_used <= (max_cards):
                    max_sum = max(
                        max_sum,
                        calculate_row_sum(board[row_index], mask)
                        + dsf(row_index + 1, mask, cards_used + bits),
                    )
            memo[row_index][previous_mask] = max_sum
        return memo[row_index][previous_mask]

    return int(dsf(0, 0, 0))
