import random
from copy import deepcopy
from dataclasses import dataclass

from src.dp.bottom_up import mwis_bottom_up
from src.greedy.board_state import BoardState, Tile


@dataclass
class RegionFixContext:
    state: BoardState
    first_row: int
    last_row: int
    max_cards: int

    @property
    def region(self) -> list[list[int]]:
        return self.state.board[self.first_row : self.last_row]

    @property
    def n_columns(self) -> int:
        return self.state.m


class FixLocalRegions:
    def __init__(self, region_size: int, rng: random.Random) -> None:
        self.region_size = region_size
        self.rng = rng

    def __call__(self, state: BoardState, max_cards: int) -> BoardState:
        cloned = deepcopy(state)
        self._fix_region(cloned, max_cards)
        return cloned

    def _fix_region(self, state: BoardState, max_cards: int) -> None:
        first_row, last_row = self._get_region_boundaries(state)
        context = RegionFixContext(state, first_row, last_row, max_cards)
        initial_mask, final_mask = self._get_boundary_masks(context)
        selection_delta = self._calculate_selection_delta(context)
        self._clear_region_selection(context)
        result, _ = mwis_bottom_up(
            context.region, max_cards + selection_delta, initial_mask, final_mask
        )
        _, fixed_region = result
        self._select_found_tiles(fixed_region, context)

    def _get_region_boundaries(self, state: BoardState) -> tuple[int, int]:
        first_row = self.rng.randint(0, state.n - 1)
        last_row = min(first_row + self.region_size, state.n)
        return first_row, last_row

    def _get_boundary_masks(self, context: RegionFixContext) -> tuple[int, int]:
        initial_mask, final_mask = 0, 0
        if (row := context.first_row - 1) >= 0:
            initial_mask = context.state.get_mask_from_row(row)
        if (row := context.last_row) < context.state.n:
            final_mask = context.state.get_mask_from_row(row)
        return initial_mask, final_mask

    def _clear_region_selection(self, context: RegionFixContext) -> None:
        for i in range(context.first_row, context.last_row):
            for j in range(context.state.m):
                if context.state.selection_grid[i][j]:
                    context.state.unselect_tile((i, j))

    def _calculate_selection_delta(self, context: RegionFixContext) -> int:
        n_selected_in_region = 0
        for i in range(context.first_row, context.last_row):
            for j in range(context.n_columns):
                if context.state.selection_grid[i][j]:
                    n_selected_in_region += 1
        return n_selected_in_region - context.state.count_selected_tiles()

    def _select_found_tiles(self, region: list[int], context: RegionFixContext) -> None:
        tiles = self._convert_masks_to_tiles(region, context.first_row, context.state.m)
        for tile in tiles:
            context.state.select_tile(tile)

    def _convert_masks_to_tiles(
        self, masks: list[int], first_row: int, n_columns: int
    ) -> list[Tile]:
        tiles: list[Tile] = []
        for i, mask in enumerate(masks):
            for j in range(n_columns):
                if mask >> (n_columns - 1 - j) & 1:
                    tiles.append((first_row + i, j))
        return tiles
