from pathlib import Path

from src.experiment.config import PHASES

SEED = 42


def main() -> None:
    # algo_configs = AlgorithmConfig.get_default_configs()
    # for phase in PHASES:
    #     runner_config = RunnerConfig(
    #         phase, algo_configs, Path("results") / Path(f"{phase.name}"), SEED
    #     )
    #     runner = ExperimentRunner(runner_config)
    #     runner.run_parallel()
    for phase in PHASES:
        nondeterministic_csv_path = (
            Path("results") / Path(f"{phase.name}") / Path("tables/greedy.csv")
        )
        deterministic_csv_path = Path("results") / Path(f"{phase.name}") / Path("tables/astar.csv")
        output_csv_path = Path("results") / Path(f"{phase.name}") / Path("tables/greedy.csv")

        maker = NonDete(nondeterministic_csv_path, deterministic_csv_path, output_csv_path)
        maker.make_results()


if __name__ == "__main__":
    main()
