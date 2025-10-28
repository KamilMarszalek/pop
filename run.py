from dataclasses import dataclass
from random import randint

from dp.top_down import mwis_top_down


@dataclass
class Range:
    low: int
    high: int


def generate_board(rows: int, columns: int, r: Range) -> list[list[int]]:
    return [[randint(r.low, r.high) for _ in range(columns)] for _ in range(rows)]


N_COLUMNS = 4
N_ROWS = 500
LIMITS = Range(-10, 10)


def main() -> None:
    board = generate_board(N_ROWS, N_COLUMNS, LIMITS)
    result = mwis_top_down(board, 1)
    print(f"Board: {board}")
    print(f"Total sum: {result}")


if __name__ == "__main__":
    main()
