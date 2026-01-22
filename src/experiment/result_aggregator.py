from pathlib import Path

import pandas as pd

from src.experiment.config import PHASES, AlgorithmConfig


class ResultAggregator:
    def __init__(self, input_csv_path: Path, output_csv_path: Path) -> None:
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path

    def aggregate_results(self, groupby_columns: list, agg_columns: dict) -> None:
        df = pd.read_csv(self.input_csv_path)
        aggregated_df = df.groupby(groupby_columns).agg(agg_columns).reset_index()

        aggregated_df.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in aggregated_df.columns.values
        ]
        if "value_mean_mean" in aggregated_df.columns:
            aggregated_df = aggregated_df.rename(
                columns={
                    "value_mean_mean": "value_mean",
                    "value_mean_std": "value_std",
                    "time_mean_mean": "time_mean",
                    "time_mean_std": "time_std",
                }
            )

        aggregated_df["board_config_id"] = aggregated_df["board_config_id_"]
        aggregated_df = aggregated_df.drop(
            columns=[col for col in aggregated_df.columns if col.endswith("_")]
        )
        aggregated_df.to_csv(self.output_csv_path, index=False)


def aggregate_results() -> None:
    for phase in PHASES:
        for config in AlgorithmConfig.get_default_configs():
            input_csv_path = (
                Path("results")
                / Path(f"{phase.name}")
                / Path("tables")
                / Path(f"{config.name}.csv")
            )
            output_csv_path = (
                Path("results")
                / Path(f"{phase.name}")
                / Path("tables")
                / Path(f"{config.name}_aggregated.csv")
            )

            aggregator = ResultAggregator(input_csv_path, output_csv_path)
            if config.name == "greedy":
                groupby_columns = ["board_config_id"]
                agg_columns = {"value_mean": ["mean", "std"], "time_mean": ["mean", "std"]}

            else:
                groupby_columns = ["board_config_id"]
                agg_columns = {"value": ["mean", "std"], "time": ["mean", "std"]}
            aggregator.aggregate_results(groupby_columns, agg_columns)


if __name__ == "__main__":
    aggregate_results()
