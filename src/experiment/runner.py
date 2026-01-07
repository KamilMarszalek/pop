import random
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.util.types import Board, MWISSolver


@dataclass
class BoardConfig:
    n_rows: int
    n_columns: int
    low: int
    high: int

    def generate(self, rng: random.Random) -> "BoardInstance":
        board = [
            [rng.randint(self.low, self.high) for _ in range(self.n_columns)]
            for _ in range(self.n_rows)
        ]
        return BoardInstance(board, self)


@dataclass
class BoardInstance:
    board: Board
    config: BoardConfig


@dataclass
class AlgorithmConfig:
    solver: MWISSolver
    name: str
    param_grid: dict[str, list[Any]] | None = None
    is_deterministic: bool = True

    def get_configurations(self) -> Iterator[dict[str, Any]]:
        if self.param_grid is None:
            yield {}
        else:
            keys = list(self.param_grid.keys())
            values = list(self.param_grid.values())
            for combination in product(*values):
                yield dict(zip(keys, combination))


@dataclass
class RunnerConfig:
    board_configs: list[BoardConfig]
    algorithms_configs: list[AlgorithmConfig]
    max_card_percents: list[float]
    seed: int
    n_repetitions: int
    n_workers: int | None = None
    output_path: Path = Path("results")


class ExperimentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.board_configs = config.board_configs
        self.algo_configs = config.algorithms_configs
        self.max_card_percents = config.max_card_percents
        self.rng = random.Random(config.seed)
        self.n_repetitions = config.n_repetitions
        self.n_workers = config.n_workers

        config.output_path.mkdir(exist_ok=True, parents=True)
        self.output_path = config.output_path

    def run_parallel(self) -> None:
        boards = [b_cfg.generate(self.rng) for b_cfg in self.board_configs]
        tasks: list[tuple[AlgorithmConfig, BoardInstance, int, dict[str, Any]]] = []

        for algo in self.algo_configs:
            for board in boards:
                for max_cards_percent in self.max_card_percents:
                    max_cards = max(
                        1, int(board.config.n_rows * board.config.n_columns * max_cards_percent)
                    )
                    param_grid = algo.get_configurations()
                    for param in param_grid:
                        tasks.append((algo, board, max_cards, param))

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures: list[Future[dict[str, Any]]] = []
            for algo, board, max_cards, params in tasks:
                if not algo.is_deterministic:
                    params["rng"] = self.rng
                futures.append(
                    executor.submit(self._run_single_experiment, algo, board, max_cards, params)
                )
            results = [f.result() for f in futures]

        self._save_results(results)

    def _run_single_experiment(
        self, algo: AlgorithmConfig, board: BoardInstance, max_cards: int, params: dict[str, Any]
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "algo": algo.name,
            "n_rows": board.config.n_rows,
            "n_columns": board.config.n_columns,
            "limit_low": board.config.low,
            "limit_high": board.config.high,
        }
        if algo.is_deterministic:
            result, elapsed = algo.solver(board.board, max_cards, **params)
            value, _ = result
            res: dict[str, Any] = {"value": value, "time": elapsed}
            base.update(params)
            base.update(res)
            return base

        else:
            times: list[float] = []
            values: list[int] = []
            for _ in range(self.n_repetitions):
                result, elapsed = algo.solver(board.board, max_cards, **params)
                value, _ = result
                times.append(elapsed)
                values.append(value)
            results: dict[str, Any] = {
                "value_mean": np.mean(values),
                "value_std": np.std(values),
                "value_min": np.min(values),
                "value_max": np.max(values),
                "time_mean": np.mean(times),
                "time_std": np.std(times),
            }
            params.pop("rng")
            base.update(params)
            base.update(results)
            return base

    def _save_results(self, results: list[dict[str, Any]]) -> None:
        df = pd.DataFrame(results)
        for algo in self.algo_configs:
            partial = df[df["algo"] == algo.name]
            partial = partial.dropna(axis=1, how="all")
            partial.to_csv(self.output_path / f"{algo.name}.csv", index=False)
