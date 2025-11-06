import random
from typing import Callable

from util.types import Board

type HeuristicFunc = Callable[[Tile], int] | None
type Tile = tuple[int, int]


class BoardState:
    def __init__(self, board: Board) -> None:
        self.n: int = len(board)
        self.m: int = len(board[0])
        self.board: Board = board
        self.selection_grid: list[list[bool]] = [
            [False for _ in range(self.m)] for _ in range(self.n)
        ]
        self.selected_tiles: set[Tile] = set()

    def evaluate_sum(self) -> int:
        return sum(
            self.board[i][j]
            for j in range(self.m)
            for i in range(self.n)
            if self.selection_grid[i][j]
        )

    def greedy_fill(self, max_cards: int, h_func: HeuristicFunc = None) -> None:
        while (tile := self._find_best_isolated_tile(h_func)) and len(
            self.selected_tiles
        ) < max_cards:
            self._select_tile(tile)

    def select_random_positive_tile(self) -> Tile | None:
        tiles = [(i, j) for j in range(self.m) for i in range(self.n)]
        random.shuffle(tiles)
        for tile in tiles:
            row, column = tile
            if (
                not self.selection_grid[row][column]
                and self.board[row][column] > 0
                and self._can_tile_be_selected(tile)
            ):
                self._select_tile(tile)
                return tile
        return None

    def get_number_of_selected_tiles(self) -> int:
        return len(self.selected_tiles)

    def _select_tile(self, tile: Tile) -> None:
        row, column = tile
        self.selection_grid[row][column] = True
        self.selected_tiles.add(tile)

    def _can_tile_be_selected(self, tile: Tile) -> bool:
        row, column = tile
        neighbors = self._neighbors(tile)
        return (
            not any(self.selection_grid[i][j] for i, j in neighbors)
            and not self.selection_grid[row][column]
        )

    def _neighbors(self, tile: Tile) -> list[Tile]:
        row, column = tile
        result: list[Tile] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in directions:
            new_row, new_column = row + x, column + y
            if 0 <= new_row < self.n and 0 <= new_column < self.m:
                result.append((new_row, new_column))
        return result

    def _find_best_isolated_tile(self, h_func: HeuristicFunc = None) -> Tile | None:
        scoring = h_func or self._default_scoring
        best_score, best_tile = 0, None
        for i in range(self.n):
            for j in range(self.m):
                tile = (i, j)
                if not self._can_tile_be_selected(tile):
                    continue
                score = scoring(tile)
                if score > best_score:
                    best_score = score
                    best_tile = tile
        return best_tile

    def _default_scoring(self, tile: Tile) -> int:
        row, column = tile
        return self.board[row][column] // (1 + len(self._neighbors(tile)))
