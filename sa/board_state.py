from util.types import Board

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

    def neighbors(self, tile: Tile) -> list[Tile]:
        row, col = tile
        result: list[Tile] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < self.n and 0 <= new_col < self.m:
                result.append((new_row, new_col))
        return result

    def can_tile_be_selected(self, tile: Tile) -> bool:
        row, col = tile
        neighbors = self.neighbors(tile)
        return (
            not any(self.selection_grid[i][j] for i, j in neighbors)
            and not self.selection_grid[row][col]
        )

    def select_tile(self, tile: Tile) -> None:
        row, col = tile
        self.selection_grid[row][col] = True
        self.selected_tiles.add(tile)

    def unselect_tile(self, tile: Tile) -> None:
        row, col = tile
        self.selection_grid[row][col] = False
        self.selected_tiles.remove(tile)

    def count_selected_tiles(self) -> int:
        return len(self.selected_tiles)

    def get_mask_from_row(self, row_index: int) -> int:
        mask = 0
        selection_row = self.selection_grid[row_index]
        for tile in selection_row:
            mask = (mask << 1) | int(tile)
        return mask
