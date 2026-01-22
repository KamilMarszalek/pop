from pathlib import Path

import pandas as pd


class NonDeterministicResultMaker:
    def __init__(
        self, nondeterministic_csv_path: Path, deterministic_csv_path: Path, output_csv_path: Path
    ) -> None:
        self.nondeterministic_csv_path = nondeterministic_csv_path
        self.deterministic_csv_path = deterministic_csv_path
        self.output_csv_path = output_csv_path

    def make_results(self) -> None:
        non_deterministic_df = pd.read_csv(self.nondeterministic_csv_path)
        deterministic_df = pd.read_csv(
            self.deterministic_csv_path, usecols=["board_id", "value", "time"]
        )

        merged_df = pd.merge(non_deterministic_df, deterministic_df, on="board_id")
        merged_df["approximation_ratio"] = merged_df["value_mean"] / merged_df["value"]
        merged_df["time_ratio"] = merged_df["time_mean"] / merged_df["time"]
        merged_df = merged_df.drop(columns=["value", "time"])
        merged_df.to_csv(self.output_csv_path, index=False)


if __name__ == "__main__":
    non_deterministic_csv_path = Path("results/scaling-v1/tables/greedy.csv")
    deterministic_csv_path = Path("results/scaling-v1/tables/astar.csv")
    maker = NonDeterministicResultMaker(
        non_deterministic_csv_path, deterministic_csv_path, Path("greedy_enriched.csv")
    )
    maker.make_results()
