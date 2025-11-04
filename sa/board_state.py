from dataclasses import dataclass
from typing import Callable, override

from util.types import Board

type HeuristicFunc = Callable[[Tile], int] | None


@dataclass
class Tile:
    weight: int
    row: int
    column: int
    is_occupied: bool

    @override
    def __hash__(self) -> int:
        return hash((self.row, self.column))


class BoardState:
    def __init__(self, board: Board) -> None:
        self.n: int = len(board)
        self.m: int = len(board[0])
        self.board: list[list[Tile]] = [
            [Tile(board[i][j], i, j, False) for j in range(self.m)]
            for i in range(self.n)
        ]
        self.occupied_tiles: set[Tile] = set()

    def evaluate_sum(self) -> int:
        return sum(
            self.board[i][j].weight
            for j in range(self.m)
            for i in range(self.n)
            if self.board[i][j].is_occupied
        )

    def greedy_fill(self, max_cards: int, h_func: HeuristicFunc = None) -> None:
        while (tile := self._find_best_isolated_tile(h_func)) and len(
            self.occupied_tiles
        ) < max_cards:
            self.occupied_tiles.add(tile)
            tile.is_occupied = True

    def can_tile_be_selected(self, tile: Tile) -> bool:
        neighbors = self._neighbors(tile)
        return not any(n.is_occupied for n in neighbors) and not tile.is_occupied

    def _neighbors(self, tile: Tile) -> list[Tile]:
        result: list[Tile] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in directions:
            new_row, new_column = tile.row + x, tile.column + y
            if 0 <= new_row < self.n and 0 <= new_column < self.m:
                result.append(self.board[new_row][new_column])
        return result

    def _find_best_isolated_tile(self, h_func: HeuristicFunc = None) -> Tile | None:
        scoring = h_func or self._default_scoroing
        best_score, best_tile = 0, None
        for i in range(self.n):
            for j in range(self.m):
                tile = self.board[i][j]
                if not self.can_tile_be_selected(tile):
                    continue
                score = scoring(tile)
                if score > best_score:
                    best_score = score
                    best_tile = tile
        return best_tile

    def _default_scoroing(self, tile: Tile) -> int:
        return tile.weight // (1 + len(self._neighbors(tile)))
