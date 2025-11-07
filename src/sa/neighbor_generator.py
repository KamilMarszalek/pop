import random
from copy import deepcopy
from typing import Protocol

from src.dp.bottom_up import mwis_bottom_up
from src.sa.board_state import BoardState, Tile

type MaskSize = tuple[int, int]


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
            row, col = n
            if cloned.can_tile_be_selected(n) and state.board[row][col] > 0:
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
        self, state: BoardState, tile: Tile, radius: int = 3
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


class FixLocalRows:
    def __call__(self, state: BoardState, max_cards: int) -> BoardState:
        cloned = deepcopy(state)
        first_row = random.randint(0, cloned.n - 1)
        self._fix_region(cloned, max_cards, first_row)
        return cloned

    def _fix_region(
        self, state: BoardState, max_cards: int, first_row: int, n_rows: int = 4
    ) -> None:
        last_row = min(first_row + n_rows, state.n)
        region = state.board[first_row:last_row]
        region_selection_grid = state.selection_grid[first_row:last_row]
        x = sum(1 for t in region_selection_grid if t) - state.count_selected_tiles()
        for i in range(first_row, last_row):
            for j in range(state.m):
                if state.selection_grid[i][j]:
                    state.unselect_tile((i, j))

        inital_mask = (
            state.get_mask_from_row(first_row - 1) if first_row - 1 >= 0 else 0
        )
        final_mask = state.get_mask_from_row(last_row) if last_row < state.n else 0
        _, fixed_region = mwis_bottom_up(
            region,
            max_cards + x,
            inital_mask,
            final_mask,
        )
        tiles = self._convert_masks_to_tiles(fixed_region, first_row, state.m)
        for tile in tiles:
            state.select_tile(tile)

    def _convert_masks_to_tiles(
        self, masks: list[int], first_row: int, n_columns: int
    ) -> list[Tile]:
        tiles: list[Tile] = []
        for i, mask in enumerate(masks):
            for j in range(n_columns):
                if mask >> (n_columns - 1 - j) & 1:
                    tiles.append((first_row + i, j))
        return tiles
