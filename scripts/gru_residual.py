from __future__ import annotations

import pandas as pd
import numpy as np

from scripts.gru_benchmark import run_gru_benchmark


MODEL_KEY = "gru_residual"
DISPLAY_NAME = "GRU Residual"

RESIDUAL_HIDDEN_SIZE = 96
RESIDUAL_DROPOUT = 0.10
RESIDUAL_BATCH_SIZE = 512
RESIDUAL_EPOCHS = 50
RESIDUAL_LEARNING_RATE = 8e-4
RESIDUAL_WEIGHT_DECAY = 1e-5
RESIDUAL_USE_LOG1P = False
RESIDUAL_TARGET_MODE = "residual"
RESIDUAL_TARGET_SCALE_MODE = "global"
RESIDUAL_LOSS_NAME = "huber"
RESIDUAL_SAMPLE_WEIGHT_MODE = "none"

#pass through to gru_benchmark.py
def run_residual_gru_benchmark(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_meta: pd.DataFrame,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    eval_meta: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    *,
    target_column: str = "new_cases",
    random_seed: int = 42,
) -> dict[str, object]:
    return run_gru_benchmark(
        model_key=MODEL_KEY,
        display_name=DISPLAY_NAME,
        train_X=train_X,
        train_y=train_y,
        train_meta=train_meta,
        eval_X=eval_X,
        eval_y=eval_y,
        eval_meta=eval_meta,
        feature_columns=feature_columns,
        target_column=target_column,
        random_seed=random_seed,
        hidden_size=RESIDUAL_HIDDEN_SIZE,
        dropout=RESIDUAL_DROPOUT,
        batch_size=RESIDUAL_BATCH_SIZE,
        epochs=RESIDUAL_EPOCHS,
        learning_rate=RESIDUAL_LEARNING_RATE,
        weight_decay=RESIDUAL_WEIGHT_DECAY,
        use_log1p=RESIDUAL_USE_LOG1P,
        target_mode=RESIDUAL_TARGET_MODE,
        target_scale_mode=RESIDUAL_TARGET_SCALE_MODE,
        loss_name=RESIDUAL_LOSS_NAME,
        sample_weight_mode=RESIDUAL_SAMPLE_WEIGHT_MODE,
    )
