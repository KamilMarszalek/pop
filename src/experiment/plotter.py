import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plotter:
    phase: str
    results_path: Path = Path("results")
    output_path: Path = Path("plots")

    def __init__(self, phase: str, output_path: Path = Path("plots")) -> None:
        self.name = phase
        self.results_path = Path("results") / self.name
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def create_time_dimension_plot(
        self, target_config_id: int, max_cards_percent: float, greedy_params: dict[str, Any]
    ) -> None:
        selected = self._get_dimensions(target_config_id)
        frames: list[pd.DataFrame] = []
        files = ["astar.csv", "dynamic-bottom-up.csv", "dynamic-top-down.csv", "greedy.csv"]
        for file in files:
            df = pd.read_csv(self.results_path / "tables" / file)  # pyright: ignore
            df = df[df["board_config_id"].isin(selected.keys())]
            df = df[df["max_cards_percent"] == max_cards_percent]
            df["dimension"] = df["board_config_id"].map(selected)
            df["algo"] = df["algo"].astype(str)
            if "time" not in df.columns:
                df = self._handle_greedy(df, greedy_params)

            df["time"] = df["time"] * 100
            frames.append(df[["algo", "dimension", "time"]])

        agg = (
            pd.concat(frames, ignore_index=True)
            .groupby(["algo", "dimension"], as_index=False)  # pyright: ignore
            .agg(time_mean=("time", "mean"), time_std=("time", "std"))
            .sort_values(["algo", "dimension"])
            .reset_index(drop=True)
        )

        sns.set_theme(style="whitegrid", palette="colorblind")
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=agg, x="dimension", y="time_mean", hue="algo", marker="o")
        for algo in agg["algo"].unique():
            subset = agg.loc[agg["algo"] == algo, :]
            plt.errorbar(  # pyright: ignore
                subset["dimension"],
                subset["time_mean"],
                yerr=subset["time_std"],
                fmt="none",
                capsize=3,
            )

        plt.xlabel("Dimension (n_rows)")
        plt.ylabel("Time (ms)")
        plt.title("Runtime vs Dimension by Algorithm")
        plt.savefig(
            self.output_path / f"{self.name}-dim-time-{int(max_cards_percent * 100)}", dpi=200
        )

    def create_time_cards_percent_plot(
        self, target_config_id: int, greedy_params: dict[str, Any]
    ) -> None:
        frames: list[pd.DataFrame] = []
        for csv in (self.results_path / "tables").glob("*.csv"):
            df = pd.read_csv(csv)  # pyright: ignore
            df = df[df["board_config_id"] == target_config_id]
            if "time" not in df.columns:
                df = self._handle_greedy(df, greedy_params)
            df["time"] = df["time"] * 100
            frames.append(df[["algo", "max_cards_percent", "time"]])

        agg = (
            pd.concat(frames, ignore_index=True)  # pyright: ignore
            .groupby(["algo", "max_cards_percent"], as_index=False)
            .agg(time_mean=("time", "mean"), time_std=("time", "std"))
            .sort_values(["algo", "max_cards_percent"])
            .reset_index(drop=True)
        )

        sns.set_theme(style="whitegrid", palette="colorblind")
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=agg, x="max_cards_percent", y="time_mean", hue="algo", marker="o")
        plt.xlabel("Avaliable card (% of board size)")
        plt.ylabel("Time (ms)")
        plt.title("Runtime vs avaliable cards")
        plt.savefig(self.output_path / f"{self.name}-cards-time", dpi=200)

    def _get_dimensions(self, target_config_id: int) -> dict[int, int]:
        with open(self.results_path / "board_configs" / "board_configs.json") as jsonfile:
            configs: list[dict[str, Any]] = json.load(jsonfile)

        target_config = next(cfg for cfg in configs if cfg["board_config_id"] == target_config_id)

        return {
            cfg["board_config_id"]: cfg["n_rows"]
            for cfg in configs
            if cfg["distribution"] == target_config["distribution"]
        }

    def _handle_greedy(self, df: pd.DataFrame, greedy_params: dict[str, Any]) -> pd.DataFrame:
        df = df.loc[
            (df["region_percent_size"] == greedy_params["region_percent_size"])
            & (df["n_iter"] == greedy_params["n_iter"])
        ].copy()  # pyright: ignore
        df["time"] = df["time_mean"]
        return df  # pyright: ignore


if __name__ == "__main__":
    plotter1 = Plotter("scaling")
    plotter1.create_time_dimension_plot(0, 0.25, {"n_iter": 200, "region_percent_size": 0.05})
    plotter1.create_time_dimension_plot(0, 1.0, {"n_iter": 200, "region_percent_size": 0.05})
