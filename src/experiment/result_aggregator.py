from pathlib import Path

import pandas as pd

from src.experiment.config import PHASES, AlgorithmConfig


class ResultAggregator:
    def __init__(self, input_csv_path: Path, output_csv_path: Path) -> None:
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path

    def aggregate_dataframe(self, df: pd.DataFrame, groupby_columns: list, agg_columns: dict) -> pd.DataFrame:
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

        if "board_config_id_" in aggregated_df.columns:
            aggregated_df["board_config_id"] = aggregated_df["board_config_id_"]
        aggregated_df = aggregated_df.drop(
            columns=[col for col in aggregated_df.columns if col.endswith("_")]
        )
        return aggregated_df

    def aggregate_results(self, groupby_columns: list, agg_columns: dict) -> None:
        df = pd.read_csv(self.input_csv_path)
        aggregated_df = self.aggregate_dataframe(df, groupby_columns, agg_columns)
        aggregated_df.to_csv(self.output_csv_path, index=False)


def _format_max_cards_percent(max_cards_percent: float) -> str:
    percent = round(float(max_cards_percent) * 100, 6)
    if abs(percent - round(percent)) < 1e-6:
        return f"{int(round(percent))}pct"
    trimmed = f"{percent:.3f}".rstrip("0").rstrip(".")
    return f"{trimmed.replace('.', 'p')}pct"


def _resolve_agg_columns(df: pd.DataFrame) -> dict:
    if {"value_mean", "time_mean"}.issubset(df.columns):
        return {"value_mean": ["mean", "std"], "time_mean": ["mean", "std"]}
    if {"value", "time"}.issubset(df.columns):
        return {"value": ["mean", "std"], "time": ["mean", "std"]}
    raise KeyError("Missing value/time columns for aggregation.")


def aggregate_results() -> None:
    for phase in PHASES:
        for config in AlgorithmConfig.get_default_configs():
            results_path = Path("results") / Path(f"{phase.name}") / Path("tables")
            input_csv_path = results_path / Path(f"{config.name}.csv")
            df = pd.read_csv(input_csv_path)
            agg_columns = _resolve_agg_columns(df)

            if "max_cards_percent" not in df.columns:
                output_csv_path = results_path / Path(f"{config.name}_aggregated.csv")
                aggregator = ResultAggregator(input_csv_path, output_csv_path)
                aggregated_df = aggregator.aggregate_dataframe(df, ["board_config_id"], agg_columns)
                aggregated_df.to_csv(output_csv_path, index=False)
                continue

            max_cards_percents = sorted(df["max_cards_percent"].dropna().unique())
            if not max_cards_percents:
                output_csv_path = results_path / Path(f"{config.name}_aggregated.csv")
                aggregator = ResultAggregator(input_csv_path, output_csv_path)
                aggregated_df = aggregator.aggregate_dataframe(df, ["board_config_id"], agg_columns)
                aggregated_df.to_csv(output_csv_path, index=False)
                continue

            for max_cards_percent in max_cards_percents:
                filtered_df = df[df["max_cards_percent"] == max_cards_percent]
                percent_suffix = _format_max_cards_percent(max_cards_percent)
                output_csv_path = results_path / Path(
                    f"{config.name}_{percent_suffix}_aggregated.csv"
                )
                aggregator = ResultAggregator(input_csv_path, output_csv_path)
                aggregated_df = aggregator.aggregate_dataframe(
                    filtered_df, ["board_config_id"], agg_columns
                )
                aggregated_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    aggregate_results()
