import random
from copy import deepcopy
from typing import Callable

from sa.board_state import BoardState

type NeighborSelectionFunc = Callable[[BoardState], BoardState]


def remove_and_select(state: BoardState) -> BoardState:
    cloned_state = deepcopy(state)
    tile = random.choice(list(cloned_state.selected_tiles))
    row, column = tile
    cloned_state.selected_tiles.remove(tile)
    cloned_state.selection_grid[row][column] = False
    directions = [
        (i, j) for j in range(-2, 3) for i in range(-2, 3) if not (i == 0 and j == 0)
    ]
    random.shuffle(directions)
    for i, j in directions:
        new_row, new_column = row + i, column + j
        if 0 <= new_row < cloned_state.n and 0 <= new_column < cloned_state.m:
            tile = (new_row, new_column)
            if cloned_state.can_tile_be_selected(tile):
                cloned_state.selected_tiles.add(tile)
                cloned_state.selection_grid[new_row][new_column] = True
                return cloned_state
    cloned_state.selected_tiles.add(tile)
    cloned_state.selection_grid[row][column] = True
    return cloned_state
