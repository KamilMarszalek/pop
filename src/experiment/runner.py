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

    def generate(self, id: int, seed: int) -> "BoardInstance":
        rng = random.Random(seed)
        board = [
            [rng.randint(self.low, self.high) for _ in range(self.n_columns)]
            for _ in range(self.n_rows)
        ]
        return BoardInstance(id, seed, board, self)


@dataclass
class BoardInstance:
    id: int
    seed: int
    board: Board
    config: BoardConfig

    @property
    def size(self) -> int:
        return self.config.n_rows * self.config.n_columns


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
    boards_per_config: int
    seed: int
    n_repetitions: int
    n_workers: int | None = None
    output_path: Path = Path("results")


class ExperimentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.board_configs = config.board_configs
        self.algo_configs = config.algorithms_configs
        self.max_card_percents = config.max_card_percents
        self.boards_per_config = config.boards_per_config
        self.rng = random.Random(config.seed)
        self.n_repetitions = config.n_repetitions
        self.n_workers = config.n_workers

        config.output_path.mkdir(exist_ok=True, parents=True)
        self.output_path = config.output_path

    def run_parallel(self) -> None:
        tasks: list[tuple[AlgorithmConfig, BoardInstance, int, dict[str, Any]]] = []

        for i, b in enumerate(self.board_configs):
            for bi in range(self.boards_per_config):
                board_seed = self.rng.randint(0, 32**2 - 1)
                board = b.generate(i * self.boards_per_config + bi, board_seed)
                for max_cards_percent in self.max_card_percents:
                    for algo in self.algo_configs:
                        max_cards = max(1, int(board.size * max_cards_percent))
                        param_grid = algo.get_configurations()
                        for param in param_grid:
                            tasks.append((algo, board, max_cards, param))

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures: list[Future[dict[str, Any]]] = []
            for algo, board, max_cards, params in tasks:
                if not algo.is_deterministic:
                    algo_rng = random.Random(self.rng.randint(0, 2**32 - 1))
                    params["rng"] = algo_rng
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
            "board_id": board.id,
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
                "num_trials": self.n_repetitions,
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
