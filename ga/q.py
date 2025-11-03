from unit import Unit
from type_definitions import Board


def q(unit: "Unit", board: Board) -> int:
    reward = 0
    for mask, col_board in zip(unit.genes, board):
        for row in range(4):
            if (mask >> row) & 1:
                reward += col_board[row]
    return reward
