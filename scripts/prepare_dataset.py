from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_RAW_FILES = [
    "time_series_covid19_confirmed_US.csv",
    "time_series_covid19_deaths_US.csv",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_raw_data_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / "data" / "raw",
        project_root / "csse_covid_19_data" / "csse_covid_19_time_series",
    ]


def list_missing_raw_files(raw_dir: Path, expected_files: list[str] | tuple[str, ...]) -> list[str]:
    missing_files: list[str] = []

    for file_name in expected_files:
        file_path = raw_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    return missing_files


def find_raw_data_dir(
    project_root: Path,
    expected_files: list[str] | tuple[str, ...] = EXPECTED_RAW_FILES,
) -> Path:
    candidate_dirs = get_raw_data_candidates(project_root)

    for candidate_dir in candidate_dirs:
        missing_files = list_missing_raw_files(candidate_dir, expected_files)
        if not missing_files:
            return candidate_dir

    return candidate_dirs[0]


def get_date_columns(columns) -> list[str]:
    date_columns: list[str] = []

    for column_name in columns:
        if "/" in column_name and column_name[0].isdigit():
            date_columns.append(column_name)

    return date_columns


def load_us_timeseries(csv_path: Path, value_name: str) -> pd.DataFrame:
    header_columns = pd.read_csv(csv_path, nrows=0).columns
    date_columns = get_date_columns(header_columns)

    if not date_columns:
        raise ValueError(f"{csv_path.name} does not contain any date columns.")

    selected_columns = ["Province_State"] + date_columns
    frame = pd.read_csv(csv_path, usecols=selected_columns).copy()

    if "Province_State" not in frame.columns:
        raise ValueError(f"{csv_path.name} is missing the Province_State column.")

    grouped = frame.groupby("Province_State")[date_columns].sum(numeric_only=True).reset_index()
    long_frame = grouped.melt(
        id_vars=["Province_State"],
        value_vars=date_columns,
        var_name="date",
        value_name=value_name,
    )

    long_frame = long_frame.rename(columns={"Province_State": "state"})
    long_frame["date"] = pd.to_datetime(long_frame["date"], format="%m/%d/%y")
    long_frame = long_frame.sort_values(["state", "date"]).reset_index(drop=True)
    return long_frame


def build_state_daily_frame(raw_dir: Path) -> pd.DataFrame:
    confirmed_path = raw_dir / "time_series_covid19_confirmed_US.csv"
    deaths_path = raw_dir / "time_series_covid19_deaths_US.csv"

    confirmed = load_us_timeseries(confirmed_path, "confirmed")
    deaths = load_us_timeseries(deaths_path, "deaths")

    merged = confirmed.merge(deaths, on=["state", "date"], how="inner")
    merged = merged.sort_values(["state", "date"]).reset_index(drop=True)

    merged["new_cases"] = merged.groupby("state")["confirmed"].diff()
    merged["new_deaths"] = merged.groupby("state")["deaths"].diff()

    merged["new_cases"] = merged["new_cases"].fillna(merged["confirmed"]).clip(lower=0)
    merged["new_deaths"] = merged["new_deaths"].fillna(merged["deaths"]).clip(lower=0)
    merged["day_index"] = merged.groupby("state").cumcount()
    return merged


def prepare_dataset(project_root: Path) -> tuple[Path, pd.DataFrame]:
    raw_data_dir = find_raw_data_dir(project_root)
    missing_files = list_missing_raw_files(raw_data_dir, EXPECTED_RAW_FILES)

    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing raw data files in {raw_data_dir}: {missing_text}")

    state_daily = build_state_daily_frame(raw_data_dir)
    return raw_data_dir, state_daily


def write_state_daily_csv(state_daily: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_daily.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the state-level daily COVID-19 dataset.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=get_project_root(),
        help="Project root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to data/processed/us_state_daily.csv under the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    if args.output is None:
        output_path = project_root / "data" / "processed" / "us_state_daily.csv"
    else:
        output_path = args.output.resolve()

    raw_data_dir, state_daily = prepare_dataset(project_root)
    write_state_daily_csv(state_daily, output_path)

    print(f"Raw data directory: {raw_data_dir}")
    print(f"Wrote: {output_path}")
    print(f"Rows: {len(state_daily):,}")
    print(f"States: {state_daily['state'].nunique()}")
    print(f"Date range: {state_daily['date'].min().date()} to {state_daily['date'].max().date()}")


if __name__ == "__main__":
    main()
