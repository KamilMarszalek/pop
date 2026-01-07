from typing import Callable

from src.ga.type_definitions import Board
from src.ga.unit import Unit

type QFunc = Callable[["Unit", Board], int]


def q(unit: "Unit", board: Board) -> int:
    reward = 0
    for mask, col_board in zip(unit.genes, board):
        for row in range(4):
            if (mask >> row) & 1:
                reward += col_board[row]
    return reward
