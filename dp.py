from dataclasses import dataclass
from random import randint

type Board2D = list[list[int]]
type Memo = dict[int, dict[int, float]]


@dataclass
class Range:
    low: int
    high: int


N = 4
M = N * N


def generate_board(rows: int, columns: int, r: Range) -> Board2D:
    return [[randint(r.low, r.high) for _ in range(columns)] for _ in range(rows)]


def generate_non_adjacent_masks(size: int) -> list[int]:
    valid_masks: list[int] = []
    for mask in range(1 << size):
        if mask & (mask << 1):
            continue
        valid_masks.append(mask)
    return valid_masks


def calculate_row_sum(row: list[int], mask: int) -> int:
    sum: int = 0
    for i, number in enumerate(row):
        if mask >> (len(row) - i - 1) & 1:
            sum += number
    return sum


def initalize_memo(n_rows: int, masks: list[int]) -> Memo:
    return {i: {mask: float("-inf") for mask in masks} for i in range(n_rows)}


def mwis_dp(board: Board2D, max_cards: int) -> int:
    n_rows = len(board)
    n_columns = len(board[0])
    possible_masks = generate_non_adjacent_masks(n_columns)
    memo = initalize_memo(n_rows, possible_masks)

    def dp(row_index: int, previous_mask: int, cards_used: int) -> float:
        if row_index == n_rows or cards_used == max_cards:
            return 0
        if memo[row_index][previous_mask] == float("-inf"):
            max_sum = float("-inf")
            for mask in possible_masks:
                positive_bits = mask.bit_count()
                if not (previous_mask & mask) and positive_bits <= (
                    max_cards - cards_used
                ):
                    max_sum = max(
                        max_sum,
                        calculate_row_sum(board[row_index], mask)
                        + dp(row_index + 1, mask, cards_used + positive_bits),
                    )
            memo[row_index][previous_mask] = max_sum
        return memo[row_index][previous_mask]

    return int(dp(0, 0, 0))


def main() -> None:
    board = generate_board(1000, N, Range(-10, 10))
    result = mwis_dp(board, M)
    print(f"Board: {board}")
    print(f"Total sum: {result}")


if __name__ == "__main__":
    main()
