from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment.config import PHASES

map_column = {
    "dynamic_top_down": "Dynamic TD [s]",
    "dynamic_bottom_up": "Dynamic BU [s]",
    "astar": "A* [s]",
    "greedy": "Greedy [s]",
    "ga": "Genetic Algorithm [s]",
}


class TableBuilder:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        caption: str | None = None,
        label: str | None = None,
        column_lines: bool = True,
        header_hline: bool = True,
        right_align_cells: list[tuple[int, str]] | None = None,
        float_format: str = "%.3f",
        escape: bool | None = True,
    ) -> None:
        self.dataframe = dataframe
        self.caption = caption
        self.label = label
        self.column_lines = column_lines
        self.header_hline = header_hline
        self.right_align_cells = right_align_cells or []
        self.float_format = float_format
        self.escape = escape

    def _column_format(self) -> str:
        if self.dataframe.empty:
            return "l"
        alignments = ["r"] * len(self.dataframe.columns)
        if self.column_lines:
            return "|".join(alignments)
        return "".join(alignments)

    @staticmethod
    def _escape_underscores(value: object) -> object:
        if isinstance(value, str):
            return value.replace("_", "\\_")
        return value

    def _escape_caption(self) -> str | None:
        if not self.caption:
            return None
        return self.caption.replace("_", "\\_")

    def _format_cell_value(self, value: object) -> str:
        if pd.isna(value):
            return "NaN"
        if isinstance(value, (float, np.floating)):
            return self.float_format % value
        return str(value)

    def _prepare_dataframe(self) -> tuple[pd.DataFrame, bool]:
        df = self.dataframe.copy()
        needs_raw_latex = bool(self.right_align_cells)
        if not needs_raw_latex:
            return df, False

        rename_map = {}
        for col in df.columns:
            if isinstance(col, str):
                escaped_col = self._escape_underscores(col)
                if escaped_col != col:
                    rename_map[col] = escaped_col
        if rename_map:
            df = df.rename(columns=rename_map)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(self._escape_underscores)

        for row_idx, col_name in self.right_align_cells:
            escaped_col = rename_map.get(col_name, col_name)
            if escaped_col not in df.columns:
                raise KeyError(f"Column not found for right alignment: {col_name}")
            formatted_value = self._escape_underscores(
                self._format_cell_value(self.dataframe.at[row_idx, col_name])
            )
            df[escaped_col] = df[escaped_col].astype(object)
            df.at[row_idx, escaped_col] = f"\\multicolumn{{1}}{{r}}{{{formatted_value}}}"

        return df, True

    def _inject_table_style(self, latex_str: str) -> str:
        lines = latex_str.splitlines()
        if not lines or not lines[0].startswith("\\begin{table}"):
            return latex_str
        lines[1:1] = [
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{6pt}",
            "\\renewcommand{\\arraystretch}{1.2}",
        ]
        return "\n".join(lines)

    def _inject_header_hline(self, latex_str: str) -> str:
        if not self.header_hline:
            return latex_str
        lines = latex_str.splitlines()
        for idx, line in enumerate(lines):
            if line.strip() == "\\midrule":
                lines[idx] = "\\hline"
                break
        return "\n".join(lines)

    def to_latex(self, output_path: Path) -> None:
        latex_df, needs_raw_latex = self._prepare_dataframe()
        latex_str = latex_df.to_latex(
            index=False,
            float_format=self.float_format,
            column_format=self._column_format(),
            caption=self._escape_caption(),
            label=self.label,
            position="htbp",
            escape=False if needs_raw_latex else self.escape,
        )
        latex_str = self._inject_table_style(latex_str)
        latex_str = self._inject_header_hline(latex_str)
        with open(output_path, "w") as f:
            f.write(latex_str)


def build_tables() -> None:
    for phase in PHASES:
        results_path = Path("results") / Path(f"{phase.name}") / Path("tables")
        grouped_tables: dict[str, list[pd.DataFrame]] = {}
        for csv in results_path.glob("*_aggregated.csv"):
            df = pd.read_csv(csv)
            stem = csv.stem
            if stem.endswith("_aggregated"):
                stem = stem[: -len("_aggregated")]
            algo_name = stem
            percent_suffix = "all"
            if "_" in stem:
                algo_candidate, _, suffix = stem.rpartition("_")
                if suffix.endswith("pct"):
                    algo_name = algo_candidate
                    percent_suffix = suffix

            algo_key = algo_name.replace("-", "_")
            algo_label = map_column.get(algo_key, algo_name)
            combined_column = algo_label
            if "time_mean" in df.columns:
                df[combined_column] = df.apply(
                    lambda row: _format_mean_std(
                        row["time_mean"], row["time_std"] if "time_std" in row else np.nan, "%.3f"
                    ),
                    axis=1,
                )
                df = df.drop(
                    columns=["time_mean", "time_std", "value_mean", "value_std"], errors="ignore"
                )
            elif "time" in df.columns:
                df[combined_column] = df.apply(
                    lambda row: _format_mean_std(
                        row["time"], row["time_std"] if "time_std" in row else np.nan, "%.3f"
                    ),
                    axis=1,
                )
                df = df.drop(columns=["time", "value"], errors="ignore")

            df.rename(columns={"board_config_id": "Board Config ID"}, inplace=True)
            df = df.set_index("Board Config ID")[[combined_column]].reset_index()
            grouped_tables.setdefault(percent_suffix, []).append(df)

        for percent_suffix, dataframes in grouped_tables.items():
            if not dataframes:
                continue
            merged_df = pd.concat(
                [df.set_index("Board Config ID") for df in dataframes], axis=1
            ).reset_index()
            caption_suffix = ""
            output_suffix = ""
            if percent_suffix != "all":
                percent_label = percent_suffix[:-3].replace("p", ".")
                caption_suffix = f" ({percent_label}%)"
                output_suffix = f"_{percent_suffix}"
            caption = f"Time results - {phase.name.replace('-', ' ').title()}{caption_suffix}"
            label = f"tab:{phase.name}{output_suffix}"
            table_builder = TableBuilder(merged_df, caption=caption, label=label)
            latex_output_path = results_path / Path(f"{phase.name}{output_suffix}_table.tex")
            table_builder.to_latex(latex_output_path)

        _build_greedy_ratio_tables(results_path, phase.name)
        board_config_path = (
            Path("results") / Path(f"{phase.name}") / Path("board_configs") / "board_configs.json"
        )
        if not board_config_path.exists():
            continue
        board_configs = pd.read_json(board_config_path)
        board_configs["distribution"] = board_configs["distribution"].map(_format_distribution)
        board_configs.drop(columns=["n_columns"], inplace=True)
        board_configs.rename(
            columns={
                "board_config_id": f"Board Config ID",
                "n_rows": "Number of Rows",
                "distribution": "Distribution",
            },
            inplace=True,
        )
        caption = f"Board Configurations - {phase.name.replace('-', ' ').title()}"
        label = f"tab:board_configs_{phase.name}"
        table_builder = TableBuilder(board_configs, caption=caption, label=label, escape=False)
        latex_output_path = (
            Path("results") / Path(f"{phase.name}") / Path("tables") / Path("board_configs.tex")
        )
        table_builder.to_latex(latex_output_path)


def _format_distribution(value: object) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, dict):
        dist_type = value.get("type")
        if dist_type == "uniform":
            low = value.get("low")
            high = value.get("high")
            if low is None or high is None:
                return "$\\mathcal{U}(?)$"
            return f"$\\mathcal{{U}}({low}, {high})$"
        if dist_type == "skewed":
            low = value.get("low")
            high = value.get("high")
            ratio = value.get("negative_ratio")
            if low is None or high is None or ratio is None:
                return "skewed(?)"
            ratio_str = (
                f"{ratio:.2f}".rstrip("0").rstrip(".") if isinstance(ratio, float) else str(ratio)
            )
            return f"skewed({low}, {high}, p={ratio_str})"
    return str(value)


def _format_mean_std(
    mean: object, std: object, float_format: str, multiplier: float = 1.0
) -> str:
    if pd.isna(mean):
        return "NaN"
    mean_str = _format_scaled_value(mean, float_format, multiplier)
    if pd.isna(std):
        return mean_str
    std_str = _format_scaled_value(std, float_format, multiplier)
    return f"{mean_str} +/- {std_str}"


def _format_scaled_value(value: object, float_format: str, multiplier: float) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        scaled = float(value) * multiplier
        return float_format % scaled
    return str(value)


def _format_max_cards_percent(max_cards_percent: float) -> str:
    percent = round(float(max_cards_percent) * 100, 6)
    if abs(percent - round(percent)) < 1e-6:
        return f"{int(round(percent))}pct"
    trimmed = f"{percent:.3f}".rstrip("0").rstrip(".")
    return f"{trimmed.replace('.', 'p')}pct"


def _build_greedy_ratio_tables(results_path: Path, phase_name: str) -> None:
    greedy_csv = results_path / "greedy.csv"
    if not greedy_csv.exists():
        return
    df = pd.read_csv(greedy_csv)
    if "approximation_ratio" not in df.columns or "time_ratio" not in df.columns:
        return

    groupby_columns = ["board_config_id"]
    if "max_cards_percent" in df.columns:
        groupby_columns.append("max_cards_percent")

    aggregated_df = (
        df.groupby(groupby_columns)[["approximation_ratio", "time_ratio"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated_df.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in aggregated_df.columns.values
    ]
    if (
        "board_config_id_" in aggregated_df.columns
        and "board_config_id" not in aggregated_df.columns
    ):
        aggregated_df = aggregated_df.rename(columns={"board_config_id_": "board_config_id"})
    if (
        "max_cards_percent_" in aggregated_df.columns
        and "max_cards_percent" not in aggregated_df.columns
    ):
        aggregated_df = aggregated_df.rename(columns={"max_cards_percent_": "max_cards_percent"})

    if "max_cards_percent" in aggregated_df.columns:
        percent_values = sorted(aggregated_df["max_cards_percent"].dropna().unique())
    else:
        percent_values = ["all"]

    for percent_value in percent_values:
        if percent_value == "all":
            percent_suffix = "all"
            subset_df = aggregated_df
        else:
            percent_suffix = _format_max_cards_percent(percent_value)
            subset_df = aggregated_df[aggregated_df["max_cards_percent"] == percent_value]

        table_df = pd.DataFrame(
            {
                "Board Config ID": subset_df["board_config_id"],
                "Approximation Ratio [%]": subset_df.apply(
                    lambda row: _format_mean_std(
                        row.get("approximation_ratio_mean"),
                        row.get("approximation_ratio_std"),
                        "%.2f",
                        100.0,
                    ),
                    axis=1,
                ),
                "Time Ratio [%]": subset_df.apply(
                    lambda row: _format_mean_std(
                        row.get("time_ratio_mean"),
                        row.get("time_ratio_std"),
                        "%.2f",
                        100.0,
                    ),
                    axis=1,
                ),
            }
        )

        caption_suffix = ""
        output_suffix = ""
        if percent_suffix != "all":
            percent_label = percent_suffix[:-3].replace("p", ".")
            caption_suffix = f" ({percent_label}%)"
            output_suffix = f"_{percent_suffix}"
        caption = f"Greedy ratios - {phase_name.replace('-', ' ').title()}{caption_suffix}"
        label = f"tab:greedy_ratios_{phase_name}{output_suffix}"
        table_builder = TableBuilder(table_df, caption=caption, label=label)
        latex_output_path = results_path / Path(f"greedy_ratios{output_suffix}_table.tex")
        table_builder.to_latex(latex_output_path)


if __name__ == "__main__":
    build_tables()
