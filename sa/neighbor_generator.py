import random
from copy import deepcopy
from typing import Protocol

from sa.board_state import BoardState, Tile


class NeighborGenerator(Protocol):
    def __call__(self, state: BoardState, max_cards: int) -> BoardState: ...


class LocalNeighborGenerator:
    def __call__(self, state: BoardState, max_cards: int) -> BoardState:
        cloned = deepcopy(state)
        if (
            cloned.count_selected_tiles() < max_cards
            and (tile := self._find_unselected_positive_tile(cloned)) is not None
        ):
            cloned.select_tile(tile)
            return cloned
        tile = random.choice(list(cloned.selected_tiles))
        cloned.unselect_tile(tile)
        neighbor_tiles = self._generate_neighbor_tiles_in_range(cloned, tile)
        for n in neighbor_tiles:
            if cloned.can_tile_be_selected(n):
                cloned.select_tile(n)
                return cloned
        cloned.select_tile(tile)
        return cloned

    def _find_unselected_positive_tile(self, state: BoardState) -> Tile | None:
        tiles = [(i, j) for j in range(state.m) for i in range(state.n)]
        random.shuffle(tiles)
        for tile in tiles:
            row, col = tile
            if (
                not state.selection_grid[row][col]
                and state.board[row][col] > 0
                and state.can_tile_be_selected(tile)
            ):
                return tile
        return None

    def _generate_neighbor_tiles_in_range(
        self, state: BoardState, tile: Tile, radius: int = 2
    ) -> list[tuple[int, int]]:
        directions = [
            (i, j)
            for j in range(-radius + 1, radius)
            for i in range(-radius + 1, radius)
            if not (i == 0 and j == 0)
        ]
        random.shuffle(directions)
        row, col = tile
        return [
            (row + dx, col + dy)
            for dx, dy in directions
            if 0 <= row + dx < state.n and 0 <= col + dy < state.m
        ]
