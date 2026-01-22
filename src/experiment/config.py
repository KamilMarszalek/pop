import json
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from src.astar.astar import run_astar
from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.experiment.distribution import UniformDistribution, ValueDistribution
from src.ga.crossover import crossover
from src.ga.genetic_algorithm import run_genetic_algorithm
from src.ga.mutation import mutation
from src.ga.q import q
from src.ga.reproduction import reproduction
from src.ga.succession import elitism
from src.greedy.greedy_and_repair import greedy_and_repair
from src.util.types import Board, MWISSolver

N_COLUMNS = 4
type AlgorithmName = Literal["dynamic-top-down", "dynamic-bottom-up", "astar", "greedy", "ga"]


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
    def default_algo_config(cls, name: AlgorithmName) -> "AlgorithmConfig":
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
            case "ga":
                return cls(
                    name="ga",
                    solver=run_genetic_algorithm,
                    param_grid={
                        "q": [q],
                        "mutation": [mutation],
                        "crossover": [crossover],
                        "reproduction": [reproduction],
                        "succession": [elitism],
                        "population_count": [50],
                        "probability_of_mutation": [0.01],
                        "probability_of_crossover": [0.95],
                        "fes": [10000],
                        "num_of_best_survivors": [2],
                    },
                    is_deterministic=False,
                )
            case "greedy":
                return cls(
                    name="greedy",
                    solver=greedy_and_repair,
                    param_grid={"n_iter": [200], "region_percent_size": [0.05]},
                    is_deterministic=False,
                )

    @staticmethod
    def get_default_configs() -> list["AlgorithmConfig"]:
        names: list[AlgorithmName] = [
            "dynamic-bottom-up",
            "dynamic-top-down",
            "ga",
            "astar",
            "greedy",
        ]

        return [AlgorithmConfig.default_algo_config(name) for name in names]

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
    repetitions: int = 3

    def create_board_configs(self) -> list[BoardConfig]:
        configs: list[BoardConfig] = []
        for n_rows in self.board_heights:
            for distribution in self.distributions:
                config = BoardConfig(len(configs), n_rows, N_COLUMNS, distribution)
                configs.append(config)
        return configs


PHASES: list[ExperimentPhase] = [
    ExperimentPhase(
        name="scaling",
        board_heights=np.arange(100, 501, 50).tolist(),
        distributions=[UniformDistribution(-1000, 1000)],
        max_cards_percents=[0.25, 1.0],
        boards_per_config=5,
    ),
    ExperimentPhase(
        name="card-number",
        board_heights=[500],
        distributions=[UniformDistribution(-1000, 1000)],
        max_cards_percents=np.arange(0.1, 1.01, 0.1).tolist(),
        boards_per_config=5,
    ),
    ExperimentPhase(
        name="genetic",
        board_heights=[200],
        distributions=[UniformDistribution(-1000, 1000)],
        max_cards_percents=[0.25],
        boards_per_config=5,
    ),
]
