from src.experiment.runner import ExperimentRunner
from src.experiment.settings import default_runner_config


def main() -> None:
    runner_config = default_runner_config()
    runner = ExperimentRunner(runner_config)
    runner.run_parallel()


if __name__ == "__main__":
    main()
