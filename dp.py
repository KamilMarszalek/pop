from dataclasses import dataclass
from random import randint

type Board2D = list[list[int]]


@dataclass
class Range:
    low: int
    high: int


@dataclass
class Memo:
    current_row: list[float]
    next_row: list[float]


N = 4


def generate_board(rows: int, columns: int, r: Range) -> Board2D:
    return [[randint(r.low, r.high) for _ in range(columns)] for _ in range(rows)]


def generate_non_adjacent_masks(size: int) -> list[int]:
    valid_masks: list[int] = []
    for mask in range(1 << size):
        if mask & (mask << 1):
            continue
        valid_masks.append(mask)
    return valid_masks


def create_mask_index_map(masks: list[int]) -> dict[int, int]:
    return {mask: i for i, mask in enumerate(masks)}


def calculate_row_sum(row: list[int], mask: int) -> int:
    sum: int = 0
    for i, number in enumerate(row):
        if mask >> (len(row) - i - 1) & 1:
            sum += number
    return sum


def create_memo(masks: list[int]) -> Memo:
    n_masks = len(masks)
    current_row = [float("-inf")] * n_masks
    next_row = [0.0] * n_masks
    return Memo(current_row, next_row)


def mwis_dp(board: Board2D, max_cards: int) -> float:
    possible_masks = generate_non_adjacent_masks(len(board[0]))
    index_map = create_mask_index_map(possible_masks)
    memo = create_memo(possible_masks)

    for row_index in range(len(board) - 1, -1, -1):
        for previous_mask in possible_masks:
            if memo.current_row[index_map[previous_mask]] != float("-inf"):
                continue
            max_sum = float("-inf")
            for mask in possible_masks:
                if not (previous_mask & mask):
                    max_sum = max(
                        max_sum,
                        calculate_row_sum(board[row_index], mask)
                        + memo.next_row[index_map[mask]],
                    )
            memo.current_row[index_map[previous_mask]] = int(max_sum)
        memo.next_row = memo.current_row
        memo.current_row = [float("-inf")] * len(possible_masks)

    return memo.next_row[0]


def main() -> None:
    board = generate_board(200, N, Range(-10, 10))
    result = mwis_dp(board, 10 * N)
    print(f"Board: {board}")
    print(f"Total sum: {result}")


if __name__ == "__main__":
    main()
