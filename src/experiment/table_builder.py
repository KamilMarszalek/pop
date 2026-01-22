from pathlib import Path

import pandas as pd

from src.experiment.config import PHASES


class TableBuilder:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def to_latex(self, output_path: Path) -> None:
        latex_str = self.dataframe.to_latex(index=False)
        with open(output_path, "w") as f:
            f.write(latex_str)


def build_tables() -> None:
    for phase in PHASES:
        dataframes = []
        results_path = Path("results") / Path(f"{phase.name}") / Path("tables")
        for csv in results_path.glob("*_aggregated.csv"):
            df = pd.read_csv(csv)
            df[f"time_{csv.stem.split('_')[0]}"] = df["time_mean"]
            df[f"value_{csv.stem.split('_')[0]}"] = df["value_mean"]
            df = df.drop(columns=["time_mean", "time_std", "value_mean", "value_std"])
            dataframes.append(df)

        for df in dataframes:
            df.set_index("board_config_id", inplace=True)
            merged_df = pd.concat(dataframes, axis=1)
            merged_df.reset_index(inplace=True)

            table_builder = TableBuilder(merged_df)
            latex_output_path = results_path / Path(f"{phase.name}_table.tex")
            table_builder.to_latex(latex_output_path)


if __name__ == "__main__":
    build_tables()
