import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import override

from util.types import Board


@dataclass
class Tile:
    weight: int
    row: int
    column: int
    is_occupied: bool

    @override
    def __hash__(self) -> int:
        return hash((self.row, self.column))


class TileBoard:
    def __init__(self, board: Board) -> None:
        self.n: int = len(board)
        self.m: int = len(board[0])
        self.board: list[list[Tile]] = [
            [Tile(board[i][j], i, j, False) for j in range(self.m)]
            for i in range(self.n)
        ]
        self.occupied_tiles: set[Tile] = set()

    def neighbors(self, tile: Tile) -> list[Tile]:
        result: list[Tile] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in directions:
            new_row, new_column = tile.row + x, tile.column + y
            if 0 <= new_row < self.n and 0 <= new_column < self.m:
                result.append(self.board[new_row][new_column])
        return result

    def can_tile_be_selected(self, tile: Tile) -> bool:
        neighbors = self.neighbors(tile)
        return (
            not any(neigh.is_occupied for neigh in neighbors) and not tile.is_occupied
        )

    def find_best_isolated_tile(self) -> Tile | None:
        max_weight, best_tile = 0, None
        for i in range(self.n):
            for j in range(self.m):
                tile = self.board[i][j]
                if not self.can_tile_be_selected(tile):
                    continue
                if tile.weight / (len(self.neighbors(tile)) + 1) > max_weight:
                    max_weight = tile.weight
                    best_tile = tile
        return best_tile

    def evaluate_total_weight(self) -> int:
        return sum(
            self.board[i][j].weight
            for j in range(self.m)
            for i in range(self.n)
            if self.board[i][j].is_occupied
        )


def greedy_solution(tile_board: TileBoard, max_cards: int) -> TileBoard:
    while (tile := tile_board.find_best_isolated_tile()) and len(
        tile_board.occupied_tiles
    ) <= max_cards:
        tile_board.occupied_tiles.add(tile)
        tile.is_occupied = True
    return tile_board


def perform_local_modification(tile_board: TileBoard, tile: Tile) -> TileBoard:
    tile_board_copy = deepcopy(tile_board)
    tile_board_copy.occupied_tiles.remove(tile)
    tile.is_occupied = False
    directions = [(i, j) for j in range(-1, 2) for i in range(-1, 2)]
    random.shuffle(directions)
    for i, j in directions:
        new_row, new_colum = tile.row + i, tile.column + j
        if 0 <= new_row < tile_board_copy.n and 0 <= new_colum < tile_board_copy.m:
            new_tile = tile_board_copy.board[new_row][new_colum]
            if tile_board_copy.can_tile_be_selected(new_tile):
                tile_board_copy.occupied_tiles.add(new_tile)
                new_tile.is_occupied = True
                return tile_board_copy
    return tile_board_copy


@dataclass
class SimulatedAnnealingParams:
    n_iter: int = 1_000
    T0: float = 10.0
    T0_threshold: float = 1e-3
    cooling: float = 0.99


def simulated_annealing(
    board: Board, max_cards: int, params: SimulatedAnnealingParams
) -> int:
    x = greedy_solution(TileBoard(board), max_cards)
    x_eval = x.evaluate_total_weight()
    t = params.T0
    for _ in range(params.n_iter):
        random_occupied_tile = random.choice(list(x.occupied_tiles))
        y = perform_local_modification(x, random_occupied_tile)
        y_eval = y.evaluate_total_weight()
        if y_eval > x_eval:
            x, x_eval = y, y_eval
        elif random.uniform(0, 1) < math.exp(-abs(y_eval - x_eval) / params.T0):
            x, x_eval = y, y_eval
        t *= params.cooling
        if t < params.T0_threshold:
            break
    return x_eval
