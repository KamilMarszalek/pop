import json
import random
from dataclasses import dataclass
from itertools import count, product
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from src.astar.astar import run_astar
from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.experiment.distribution import UniformDistribution, ValueDistribution
from src.sa.greedy_and_repair import greedy_and_repair
from src.util.types import Board, MWISSolver

N_COLUMNS = 4
type AlgorithmName = Literal["dynamic-top-down", "dynamic-bottom-up", "astar", "greedy"]

_config_counter = count()
_board_counter = count()


@dataclass
class BoardConfig:
    config_id: int
    n_rows: int
    n_columns: int
    distribution: ValueDistribution

    def generate_instance(self, id: int, seed: int) -> "BoardInstance":
        rng = random.Random(seed)
        board = [
            [self.distribution.sample(rng) for _ in range(self.n_columns)]
            for _ in range(self.n_rows)
        ]

        return BoardInstance(id, seed, board, self)

    @staticmethod
    def save_board_configs(configs: list["BoardConfig"], path: Path) -> None:
        config_dicts: list[dict[str, Any]] = [
            {
                "board_config_id": c.config_id,
                "n_rows": c.n_rows,
                "n_columns": c.n_columns,
                "distribution": c.distribution.to_dict(),
            }
            for c in configs
        ]

        with open(path / "board_configs.json", "w") as json_file:
            json.dump(config_dicts, json_file, indent=4)


@dataclass
class BoardInstance:
    board_id: int
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

    @classmethod
    def _default_algo_config(cls, name: AlgorithmName) -> "AlgorithmConfig":
        match name:
            case "dynamic-top-down":
                return cls(
                    name="dynamic-top-down",
                    solver=mwis_top_down,
                    param_grid=None,
                    is_deterministic=True,
                )
            case "dynamic-bottom-up":
                return cls(
                    name="dynamic-bottom-up",
                    solver=mwis_bottom_up,
                    param_grid=None,
                    is_deterministic=True,
                )
            case "astar":
                return cls(name="astar", solver=run_astar, param_grid=None, is_deterministic=True)
            case "greedy":
                return cls(
                    name="greedy",
                    solver=greedy_and_repair,
                    param_grid={"n_iter": [200], "region_percent_size": [0.05]},
                    is_deterministic=False,
                )

    @staticmethod
    def get_default_configs() -> list["AlgorithmConfig"]:
        names: list[AlgorithmName] = ["dynamic-bottom-up", "dynamic-top-down", "astar", "greedy"]

        return [AlgorithmConfig._default_algo_config(name) for name in names]

    def get_configurations(self) -> Iterator[dict[str, Any]]:
        if self.param_grid is None:
            yield {}
        else:
            keys = list(self.param_grid.keys())
            values = list(self.param_grid.values())
            for combination in product(*values):
                yield dict(zip(keys, combination))


@dataclass
class ExperimentPhase:
    name: str
    board_heights: list[int]
    distributions: list[ValueDistribution]
    max_cards_percents: list[float]
    boards_per_config: int
    repetitions: int = 1

    def create_board_configs(self) -> list[BoardConfig]:
        configs: list[BoardConfig] = []
        for n_rows in self.board_heights:
            for distribution in self.distributions:
                config = BoardConfig(len(configs), n_rows, N_COLUMNS, distribution)
                configs.append(config)
        return configs


PHASES: list[ExperimentPhase] = [
    ExperimentPhase(
        name="card-number",
        board_heights=[500],
        distributions=[UniformDistribution(-1000, 1000)],
        max_cards_percents=np.arange(0.1, 1.01, 0.2).tolist(),
        boards_per_config=7,
    )
    # ExperimentPhase(
    #     name="comparing-greedy-with-deterministic",
    #     board_heights=np.arange(100, 501, 50).tolist(),
    #     distributions=[UniformDistribution(-1000, 1000)],
    #     max_cards_percents=[0.1],
    #     boards_per_config=5,
    # ),
]
