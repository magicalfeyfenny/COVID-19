from __future__ import annotations

import pandas as pd
import numpy as np

from scripts.gru_benchmark import run_gru_benchmark


MODEL_KEY = "gru_direct"
DISPLAY_NAME = "GRU Direct"

DIRECT_HIDDEN_SIZE = 64
DIRECT_DROPOUT = 0.10
DIRECT_BATCH_SIZE = 512
DIRECT_EPOCHS = 6
DIRECT_LEARNING_RATE = 1e-3
DIRECT_WEIGHT_DECAY = 1e-5
DIRECT_USE_LOG1P = True
DIRECT_TARGET_MODE = "raw"
DIRECT_TARGET_SCALE_MODE = "global"
DIRECT_LOSS_NAME = "mse"
DIRECT_SAMPLE_WEIGHT_MODE = "none"


def run_direct_gru_benchmark(
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
        hidden_size=DIRECT_HIDDEN_SIZE,
        dropout=DIRECT_DROPOUT,
        batch_size=DIRECT_BATCH_SIZE,
        epochs=DIRECT_EPOCHS,
        learning_rate=DIRECT_LEARNING_RATE,
        weight_decay=DIRECT_WEIGHT_DECAY,
        use_log1p=DIRECT_USE_LOG1P,
        target_mode=DIRECT_TARGET_MODE,
        target_scale_mode=DIRECT_TARGET_SCALE_MODE,
        loss_name=DIRECT_LOSS_NAME,
        sample_weight_mode=DIRECT_SAMPLE_WEIGHT_MODE,
    )
