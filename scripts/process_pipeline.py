from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EVAL_DAYS = 28
DEFAULT_LOOKBACK_DAYS = 28
DEFAULT_FORECAST_HORIZON = 7
DEFAULT_TARGET_COLUMN = "new_cases"

DEFAULT_FEATURE_COLUMNS = [
    "new_cases",
    "new_deaths",
    "cases_lag_1",
    "cases_lag_7",
    "cases_roll_mean_7",
    "cases_roll_mean_14",
    "deaths_lag_1",
    "deaths_lag_7",
    "deaths_roll_mean_7",
]

DEFAULT_BASELINE_SPECS = [
    ("Last day", "baseline_last_day"),
    ("Last week", "baseline_last_week"),
    ("7-day mean", "baseline_roll_mean_7"),
    ("14-day mean", "baseline_roll_mean_14"),
    ("Train mean by state", "baseline_train_mean_state"),
]

DEFAULT_SEQUENCE_BASELINE_SPECS = [
    ("Last observed day", "baseline_last_observed"),
    ("7-day input mean", "baseline_input_mean_7"),
    ("14-day input mean", "baseline_input_mean_14"),
    ("Train target mean by state", "baseline_train_target_mean_state"),
]


def add_basic_features(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []

    for state_name in frame["state"].unique():
        state_frame = frame[frame["state"] == state_name].copy()
        state_frame = state_frame.sort_values("date").reset_index(drop=True)

        state_frame["cases_lag_1"] = state_frame["new_cases"].shift(1)
        state_frame["cases_lag_7"] = state_frame["new_cases"].shift(7)
        state_frame["cases_roll_mean_7"] = state_frame["new_cases"].rolling(window=7).mean().shift(1)
        state_frame["cases_roll_mean_14"] = state_frame["new_cases"].rolling(window=14).mean().shift(1)

        state_frame["deaths_lag_1"] = state_frame["new_deaths"].shift(1)
        state_frame["deaths_lag_7"] = state_frame["new_deaths"].shift(7)
        state_frame["deaths_roll_mean_7"] = state_frame["new_deaths"].rolling(window=7).mean().shift(1)

        feature_frames.append(state_frame)

    result = pd.concat(feature_frames, ignore_index=True)
    result = result.sort_values(["state", "date"]).reset_index(drop=True)
    return result


def make_time_split(frame: pd.DataFrame, eval_days: int = DEFAULT_EVAL_DAYS) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = frame["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=eval_days)

    train_frame = frame[frame["date"] <= cutoff_date].copy()
    eval_frame = frame[frame["date"] > cutoff_date].copy()
    return train_frame, eval_frame, cutoff_date


def build_baseline_frame(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    baseline_frame = eval_frame.copy()

    baseline_frame["baseline_last_day"] = baseline_frame["cases_lag_1"]
    baseline_frame["baseline_last_week"] = baseline_frame["cases_lag_7"]
    baseline_frame["baseline_roll_mean_7"] = baseline_frame["cases_roll_mean_7"]
    baseline_frame["baseline_roll_mean_14"] = baseline_frame["cases_roll_mean_14"]

    train_state_mean = train_frame.groupby("state")[target_column].mean()
    baseline_frame["baseline_train_mean_state"] = baseline_frame["state"].map(train_state_mean)
    return baseline_frame


def compute_regression_metrics(frame: pd.DataFrame, actual_column: str, prediction_column: str) -> pd.DataFrame:
    clean = frame[[actual_column, prediction_column]].dropna().copy()
    if clean.empty:
        raise ValueError("No rows available to score.")

    actual = clean[actual_column].astype(float)
    predicted = clean[prediction_column].astype(float)
    error = actual - predicted
    abs_error = error.abs()

    percentage_denominator = actual.replace(0, np.nan)
    symmetric_denominator = (actual.abs() + predicted.abs()).replace(0, np.nan)
    actual_sum = actual.abs().sum()

    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    mape = (abs_error / percentage_denominator).mean() * 100.0
    smape = (200.0 * abs_error / symmetric_denominator).mean()

    if actual_sum == 0:
        wape = np.nan
    else:
        wape = abs_error.sum() / actual_sum * 100.0

    bias = error.mean()

    metrics = pd.DataFrame(
        [
            {
                "Scored Rows": len(clean),
                "MAE": mae,
                "RMSE": rmse,
                "WAPE (%)": wape,
                "MAPE (%)": mape,
                "sMAPE (%)": smape,
                "Bias": bias,
            }
        ]
    )
    return metrics


def compare_baselines(
    frame: pd.DataFrame,
    actual_column: str,
    baseline_specs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for baseline_name, prediction_column in baseline_specs:
        metric_row = compute_regression_metrics(frame, actual_column, prediction_column).iloc[0].to_dict()
        metric_row["Baseline"] = baseline_name
        rows.append(metric_row)

    score_frame = pd.DataFrame(rows)

    rank_metrics = ["MAE", "RMSE", "WAPE (%)", "MAPE (%)", "sMAPE (%)"]
    for metric_name in rank_metrics:
        score_frame[f"{metric_name} Rank"] = score_frame[metric_name].rank(method="min")

    rank_columns = [f"{metric_name} Rank" for metric_name in rank_metrics]
    score_frame["Average Rank"] = score_frame[rank_columns].mean(axis=1)

    score_frame = score_frame[
        [
            "Baseline",
            "Average Rank",
            "Scored Rows",
            "MAE",
            "RMSE",
            "WAPE (%)",
            "MAPE (%)",
            "sMAPE (%)",
            "Bias",
        ]
    ]
    score_frame = score_frame.sort_values(["Average Rank", "RMSE", "MAE", "Baseline"]).reset_index(drop=True)
    return score_frame


def compare_baselines_by_state(
    frame: pd.DataFrame,
    actual_column: str,
    baseline_specs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    state_rows: list[pd.DataFrame] = []

    for state_name in sorted(frame["state"].unique()):
        state_frame = frame[frame["state"] == state_name]
        state_scores = compare_baselines(state_frame, actual_column, baseline_specs).copy()
        state_scores.insert(0, "state", state_name)
        state_rows.append(state_scores)

    if not state_rows:
        return pd.DataFrame()

    return pd.concat(state_rows, ignore_index=True)


def summarize_best_baseline_by_state(state_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if state_scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    best_by_state = state_scores.sort_values(
        ["state", "Average Rank", "RMSE", "MAE", "Baseline"]
    ).groupby("state", as_index=False).first()

    best_counts = best_by_state["Baseline"].value_counts().rename_axis("Baseline").reset_index(
        name="Best State Count"
    )
    return best_by_state, best_counts


def build_sequence_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    clean = frame.sort_values(["state", "date"]).copy()
    clean = clean.dropna(subset=list(feature_columns) + [target_column]).reset_index(drop=True)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, object]] = []

    for state_name in clean["state"].unique():
        state_frame = clean[clean["state"] == state_name].copy()
        state_frame = state_frame.sort_values("date").reset_index(drop=True)

        max_start = len(state_frame) - lookback_days - forecast_horizon + 1
        if max_start <= 0:
            continue

        for start_index in range(max_start):
            end_index = start_index + lookback_days
            target_index = end_index + forecast_horizon - 1

            feature_window = state_frame.loc[start_index : end_index - 1, list(feature_columns)].to_numpy(
                dtype=np.float32
            )
            target_value = float(state_frame.loc[target_index, target_column])
            target_date = state_frame.loc[target_index, "date"]

            X_list.append(feature_window)
            y_list.append(target_value)
            meta_rows.append({"state": state_name, "target_date": target_date})

    if not X_list:
        raise ValueError("No sequences could be created. Check the window sizes and missing values.")

    X = np.stack(X_list)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def split_sequence_dataset_by_date(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    target_dates = pd.to_datetime(meta["target_date"])
    train_mask = (target_dates <= cutoff_date).to_numpy()
    eval_mask = (target_dates > cutoff_date).to_numpy()

    train_X = X[train_mask]
    train_y = y[train_mask]
    train_meta = meta.loc[train_mask].reset_index(drop=True)

    eval_X = X[eval_mask]
    eval_y = y[eval_mask]
    eval_meta = meta.loc[eval_mask].reset_index(drop=True)
    return train_X, train_y, train_meta, eval_X, eval_y, eval_meta


def build_sequence_baseline_frame(
    train_y: np.ndarray,
    train_meta: pd.DataFrame,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    eval_meta: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    target_feature_index = list(feature_columns).index(DEFAULT_TARGET_COLUMN)

    baseline_frame = eval_meta.copy()
    baseline_frame["actual_new_cases"] = eval_y
    baseline_frame["baseline_last_observed"] = eval_X[:, -1, target_feature_index]
    baseline_frame["baseline_input_mean_7"] = eval_X[:, -7:, target_feature_index].mean(axis=1)
    baseline_frame["baseline_input_mean_14"] = eval_X[:, -14:, target_feature_index].mean(axis=1)

    train_target_frame = train_meta.copy()
    train_target_frame["train_target"] = train_y
    train_mean_by_state = train_target_frame.groupby("state")["train_target"].mean()
    baseline_frame["baseline_train_target_mean_state"] = baseline_frame["state"].map(train_mean_by_state)
    return baseline_frame


def run_processing_pipeline(
    state_daily: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = DEFAULT_TARGET_COLUMN,
    eval_days: int = DEFAULT_EVAL_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    baseline_specs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = DEFAULT_BASELINE_SPECS,
) -> dict[str, object]:
    model_frame = add_basic_features(state_daily)
    train_frame, eval_frame, cutoff_date = make_time_split(model_frame, eval_days=eval_days)

    baseline_eval = build_baseline_frame(train_frame, eval_frame, target_column=target_column)
    baseline_scores = compare_baselines(baseline_eval, target_column, baseline_specs)
    state_baseline_scores = compare_baselines_by_state(baseline_eval, target_column, baseline_specs)
    best_baseline_by_state, best_baseline_state_counts = summarize_best_baseline_by_state(state_baseline_scores)

    X, y, meta = build_sequence_dataset(
        frame=model_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        lookback_days=lookback_days,
        forecast_horizon=forecast_horizon,
    )

    return {
        "model_frame": model_frame,
        "train_frame": train_frame,
        "eval_frame": eval_frame,
        "cutoff_date": cutoff_date,
        "baseline_eval": baseline_eval,
        "baseline_scores": baseline_scores,
        "state_baseline_scores": state_baseline_scores,
        "best_baseline_by_state": best_baseline_by_state,
        "best_baseline_state_counts": best_baseline_state_counts,
        "X": X,
        "y": y,
        "meta": meta,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "eval_days": eval_days,
        "lookback_days": lookback_days,
        "forecast_horizon": forecast_horizon,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Process the prepared COVID-19 dataset for modeling.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "processed" / "us_state_daily.csv",
        help="Input CSV path for the prepared state-level dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed" / "us_state_daily_features.csv",
        help="Output CSV path for the feature-engineered dataset.",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=DEFAULT_EVAL_DAYS,
        help="Number of days reserved for the evaluation window.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Number of lookback days per sequence sample.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON,
        help="Forecast horizon in days.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    state_daily = pd.read_csv(input_path, parse_dates=["date"])
    results = run_processing_pipeline(
        state_daily,
        eval_days=args.eval_days,
        lookback_days=args.lookback_days,
        forecast_horizon=args.forecast_horizon,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame = results["model_frame"]
    baseline_scores = results["baseline_scores"]
    X = results["X"]
    y = results["y"]

    model_frame.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(model_frame):,}")
    print(f"Sequence samples: {len(y):,}")
    print(f"X shape: {X.shape}")
    print("Baseline scores:")
    print(baseline_scores.to_string(index=False))


if __name__ == "__main__":
    main()
