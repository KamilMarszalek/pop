from pathlib import Path

from src.experiment.config import PHASES, AlgorithmConfig
from src.experiment.runner import ExperimentRunner, RunnerConfig

SEED = 42


def main() -> None:
    algo_configs = AlgorithmConfig.get_default_configs()
    for phase in PHASES:
        runner_config = RunnerConfig(
            phase, algo_configs, Path("results") / Path(f"{phase.name}"), SEED
        )
        runner = ExperimentRunner(runner_config)
        runner.run_parallel()


if __name__ == "__main__":
    main()
