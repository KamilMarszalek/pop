from typing import Callable

from src.greedy.board_state import BoardState, Tile

type TileHeuristic = Callable[[BoardState, Tile], int | float]


def greedy_fill(state: BoardState, max_cards: int, heuristic: TileHeuristic) -> None:
    while (tile := _find_best_isolated_tile(state, heuristic)) is not None and len(
        state.selected_tiles
    ) < max_cards:
        state.select_tile(tile)


def _find_best_isolated_tile(
    state: BoardState, heuristic: TileHeuristic
) -> Tile | None:
    best_score, best_tile = 0, None
    for i in range(state.n):
        for j in range(state.m):
            tile = (i, j)
            if not state.can_tile_be_selected(tile):
                continue
            score = heuristic(state, tile)
            if score > best_score:
                best_score = score
                best_tile = tile
    return best_tile


def weight(state: BoardState, tile: Tile) -> int:
    row, col = tile
    return state.board[row][col]


def weight_per_neighbors(state: BoardState, tile: Tile) -> float:
    row, col = tile
    return state.board[row][col] / (1 + len(state.neighbors(tile)))
