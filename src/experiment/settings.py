from typing import Any

from src.astar.astar import run_astar
from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.experiment.runner import AlgorithmConfig, BoardConfig, RunnerConfig
from src.sa.greedy_and_repair import greedy_and_repair

_SETTINGS: dict[str, Any] = {
    "test_board_width": 4,
    "test_board_heights": [10, 100],
    "test_value_ranges": [(0, 10), (-10, 10), (-100, 100), (-10_000, 10_000)],
    "test_max_card_selection_ratios": [0.25],
    "test_seed": 42,
    "test_num_trials": 10,
    "test_greedy": {"iterations": [100, 200], "region_sizes": [0.05, 0.10, 0.15]},
}


def _default_board_configs() -> list[BoardConfig]:
    results: list[BoardConfig] = []
    for i, (low, high) in enumerate(_SETTINGS["test_value_ranges"]):
        for j, n_rows in enumerate(_SETTINGS["test_board_heights"]):
            config = BoardConfig(
                i * len(_SETTINGS["test_board_heights"]) + j,
                n_rows,
                _SETTINGS["test_board_width"],
                low,
                high,
            )
            results.append(config)
    return results


def _default_algorithm_configs() -> list[AlgorithmConfig]:
    dp_top_down = AlgorithmConfig(
        name="dynamic-top-down", solver=mwis_top_down, param_grid=None, is_deterministic=True
    )
    dp_bottom_up = AlgorithmConfig(
        name="dynamic-bottom-up", solver=mwis_bottom_up, param_grid=None, is_deterministic=True
    )
    astar = AlgorithmConfig(name="astar", solver=run_astar, param_grid=None, is_deterministic=True)
    greedy = AlgorithmConfig(
        name="greedy",
        solver=greedy_and_repair,
        param_grid={
            "n_iter": _SETTINGS["test_greedy"]["iterations"],
            "region_percent_size": _SETTINGS["test_greedy"]["region_sizes"],
        },
        is_deterministic=False,
    )
    return [dp_top_down, dp_bottom_up, astar, greedy]


def default_runner_config() -> RunnerConfig:
    algorithms_configs = _default_algorithm_configs()
    board_configs = _default_board_configs()
    return RunnerConfig(
        board_configs,
        algorithms_configs,
        _SETTINGS["test_max_card_selection_ratios"],
        _SETTINGS["test_seed"],
        _SETTINGS["test_num_trials"],
    )
