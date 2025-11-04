import random
from copy import deepcopy
from typing import Callable

from sa.board_state import BoardState

type NeighborSelectionFunc = Callable[[BoardState], BoardState]


def remove_and_select(board_state: BoardState) -> BoardState:
    board_state_copy = deepcopy(board_state)
    tile = random.choice(list(board_state.occupied_tiles))
    board_state_copy.occupied_tiles.remove(tile)
    tile.is_occupied = False
    directions = [(i, j) for j in range(-1, 2) for i in range(-1, 2)]
    random.shuffle(directions)
    for i, j in directions:
        new_row, new_column = tile.row + i, tile.column + j
        if 0 <= new_row < board_state_copy.n and 0 <= new_column < board_state_copy.m:
            new_tile = board_state_copy.board[new_row][new_column]
            if board_state_copy.can_tile_be_selected(new_tile):
                board_state_copy.occupied_tiles.add(new_tile)
                new_tile.is_occupied = True
                return board_state_copy
    return board_state_copy
