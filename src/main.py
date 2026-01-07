import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.astar.astar import run_astar
from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.settings import MAX_CARDS_PERCENT_LIST, N_COLUMNS, N_ROWS_LIST, SEED, VALUE_RANGES_LIST
from src.util.types import Board, MWISSolver


@dataclass
class BoardData:
    board: Board
    low: int
    high: int
    n_rows: int
    n_columns: int


def _generate_boards(
    n_rows_list: list[int], limits_list: list[tuple[int, int]], rng: random.Random
) -> list[BoardData]:
    def get_board(rows: int, columns: int, low: int, high: int) -> Board:
        return [[rng.randint(low, high) for _ in range(columns)] for _ in range(rows)]

    result: list[BoardData] = []
    for n_rows in n_rows_list:
        for limits in limits_list:
            low, high = limits
            board = get_board(n_rows, N_COLUMNS, low, high)
            result.append(BoardData(board, low, high, n_rows, N_COLUMNS))

    return result


@dataclass
class SolverRun:
    name: str
    solver: MWISSolver
    params: dict[str, list[Any]] | None


def _build_param_grid(param_list: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if param_list is None:
        return [{}]
    keys = list(param_list.keys())
    values = list(param_list.values())
    grid = [dict(zip(keys, combination)) for combination in product(*values)]
    return grid


def run_experiment(
    boards: list[BoardData], max_cards_percent_list: list[float], run: SolverRun
) -> None:
    solver_results: list[dict[str, Any]] = []
    for board in boards:
        for max_cards_percent in max_cards_percent_list:
            max_cards = int((board.n_rows * board.n_columns) * max_cards_percent)
            param_grid = _build_param_grid(run.params)
            for param in param_grid:
                result, elapsed = run.solver(board.board, max_cards, **param)
                value, _ = result
                result_row: dict[str, Any] = {
                    "n_rows": board.n_rows,
                    "limit_min": board.low,
                    "limit_max": board.high,
                    "max_cards": max_cards,
                    "value": value,
                    "elapsed": elapsed,
                }
                result_row.update(param)
                solver_results.append(result_row)

    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(solver_results).to_csv(save_dir / f"{run.name}.csv")


def main() -> None:
    dp_top_down = SolverRun(name="dynamic-top-down", solver=mwis_top_down, params=None)
    dp_bottom_up = SolverRun(name="dynamic-bottom-up", solver=mwis_bottom_up, params=None)
    astar = SolverRun(name="astar", solver=run_astar, params=None)
    runs = [dp_top_down, dp_bottom_up, astar]

    rng = random.Random(SEED)
    boards = _generate_boards(N_ROWS_LIST, VALUE_RANGES_LIST, rng)

    for run in runs:
        run_experiment(boards, MAX_CARDS_PERCENT_LIST, run)


if __name__ == "__main__":
    main()
