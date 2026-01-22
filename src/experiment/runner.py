import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd

from src.experiment.config import AlgorithmConfig, BoardConfig, ExperimentPhase

type SingleExperimentTask = tuple[AlgorithmConfig, int, int, int, float, dict[str, Any]]
type SingleExperimentResult = tuple[dict[str, Any], dict[str, Any] | None]


@dataclass
class RunnerConfig:
    phase: ExperimentPhase
    algorithms_configs: list[AlgorithmConfig]
    output_path: Path
    seed: int


class ExperimentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.phase = config.phase
        self.algo_configs = config.algorithms_configs
        self.rng = random.Random(config.seed)

        self.tables_path = config.output_path / "tables"
        self.tables_path.mkdir(parents=True, exist_ok=True)

        self.logs_path = config.output_path / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self.board_config_path = config.output_path / "board_configs"
        self.board_config_path.mkdir(parents=True, exist_ok=True)

        self.log_counter: dict[str, int] = {}
        self.csv_files: dict[str, Any] = {}
        self.csv_writers: dict[str, Any] = {}

    def run_parallel(self, n_workers: int | None = None) -> None:
        if n_workers is None:
            n_workers = n_workers or (os.cpu_count() or 1)

        self.board_configs = self.phase.create_board_configs()
        BoardConfig.save_board_configs(self.board_configs, self.board_config_path)

        self._init_csv_files()

        try:
            task_args = (
                (task, self.board_configs, self.phase) for task in self._experiment_tasks()
            )

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for i, (res, log) in enumerate(
                    executor.map(_run_single_experiment, task_args, chunksize=1)
                ):
                    self._save_result_incremental(res)
                    if log is not None:
                        self._save_log_incremental(log)

                    if (i + 1) % 10 == 0:
                        print(f"Completed {i + 1} tasks...")

        finally:
            self._close_csv_files()

    def _experiment_tasks(self) -> Iterator[SingleExperimentTask]:
        for i, _ in enumerate(self.board_configs):
            for bi in range(self.phase.boards_per_config):
                board_seed = self.rng.randint(0, 2**32 - 1)
                for max_cards_percent in self.phase.max_cards_percents:
                    for algo in self.algo_configs:
                        for params in algo.get_configurations():
                            params_copy = dict(params)
                            yield algo, i, bi, board_seed, max_cards_percent, params_copy

    def _init_csv_files(self) -> None:
        for algo in self.algo_configs:
            filepath = self.tables_path / f"{algo.name}.csv"
            self.csv_files[algo.name] = open(filepath, "w", newline="")
            self.csv_writers[algo.name] = None

    def _close_csv_files(self) -> None:
        for f in self.csv_files.values():
            f.close()

    def _save_result_incremental(self, result: dict[str, Any]) -> None:
        algo_name = result["algo"]

        if self.csv_writers[algo_name] is None:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(self.csv_files[algo_name], fieldnames=fieldnames)
            writer.writeheader()
            self.csv_writers[algo_name] = writer

        self.csv_writers[algo_name].writerow(result)
        self.csv_files[algo_name].flush()

    def _save_log_incremental(self, log: dict[str, Any]) -> None:
        name = cast(str, log["algo"])

        if name not in self.log_counter:
            self.log_counter[name] = 0

        df = pd.DataFrame(log)
        df = df.drop(columns=["algo"])
        df.to_csv(self.logs_path / f"{name}-{self.log_counter[name]}.csv", index=False)

        self.log_counter[name] += 1


def _run_single_experiment(
    args: tuple[SingleExperimentTask, list[BoardConfig], ExperimentPhase],
) -> SingleExperimentResult:
    task, board_conifgs, phase = args
    algo, board_index, bi, board_seed, max_cards_percent, params = task
    board_config = board_conifgs[board_index]
    board = board_config.generate(board_index * phase.boards_per_config + bi, board_seed)
    if not algo.is_deterministic:
        algo_rng = random.Random(board_seed)
        params["rng"] = algo_rng

    base: dict[str, Any] = {
        "algo": algo.name,
        "board_id": board.id,
        "board_config_id": board.config.config_id,
        "max_cards_percent": max_cards_percent,
    }
    max_cards = max(1, int(board.size * max_cards_percent))
    if algo.is_deterministic:
        result, elapsed = algo.solver(board.board, max_cards, **params)
        assert len(result) == 2
        value, _ = result
        res: dict[str, Any] = {"value": value, "time": elapsed}
        base.update(params)
        base.update(res)
        return base, None

    else:
        times: list[float] = []
        values: list[int] = []
        logs: list[list[float]] = []
        iterations = None

        for _ in range(phase.repetitions):
            result, elapsed = algo.solver(board.board, max_cards, **params)
            assert len(result) == 3
            value, _, log = result
            iterations, log_value = log
            times.append(elapsed)
            values.append(value)
            logs.append(log_value)

        results: dict[str, Any] = {
            "num_trials": phase.repetitions,
            "value_mean": np.mean(values),
            "value_std": np.std(values),
            "value_min": np.min(values),
            "value_max": np.max(values),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
        }

        log_array = np.array(logs)
        log_info: dict[str, Any] = {
            "algo": algo.name,
            "iter": iterations,
            "eval_mean": np.mean(log_array, axis=0),
            "eval_std": np.std(log_array, axis=0),
        }

        params.pop("rng")
        base.update(params)
        base.update(results)
        return base, log_info
