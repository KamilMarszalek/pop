from pathlib import Path

from src.experiment.config import PHASES, AlgorithmConfig
from src.experiment.non_deterministic_result_maker import NonDeterministicResultMaker
from src.experiment.runner import ExperimentRunner, RunnerConfig

SEED = 42


def _csv_has_data(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    algo_configs = AlgorithmConfig.get_default_configs()
    for phase in PHASES:
        runner_config = RunnerConfig(
            phase, algo_configs, Path("results") / Path(f"{phase.name}"), SEED
        )
        runner = ExperimentRunner(runner_config)
        runner.run_parallel()
    nondeterministic_algos = [algo for algo in algo_configs if not algo.is_deterministic]
    deterministic_algos = [algo for algo in algo_configs if algo.is_deterministic]
    baseline_algo = next(
        (algo for algo in deterministic_algos if algo.name == "astar"),
        deterministic_algos[0] if deterministic_algos else None,
    )
    if baseline_algo is None or not nondeterministic_algos:
        return

    for phase in PHASES:
        tables_path = Path("results") / Path(f"{phase.name}") / Path("tables")
        deterministic_csv_path = tables_path / f"{baseline_algo.name}.csv"
        if not _csv_has_data(deterministic_csv_path):
            continue

        for algo in nondeterministic_algos:
            nondeterministic_csv_path = tables_path / f"{algo.name}.csv"
            if not _csv_has_data(nondeterministic_csv_path):
                continue
            output_csv_path = nondeterministic_csv_path
            maker = NonDeterministicResultMaker(
                nondeterministic_csv_path, deterministic_csv_path, output_csv_path
            )
            maker.make_results()


if __name__ == "__main__":
    main()
