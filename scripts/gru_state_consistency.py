from __future__ import annotations

import pandas as pd
import numpy as np

from scripts.gru_benchmark import run_gru_benchmark


MODEL_KEY = "gru_state_consistency"
DISPLAY_NAME = "GRU State Consistency"

STATE_HIDDEN_SIZE = 96
STATE_DROPOUT = 0.10
STATE_BATCH_SIZE = 512
STATE_EPOCHS = 10
STATE_LEARNING_RATE = 8e-4
STATE_WEIGHT_DECAY = 1e-5
STATE_USE_LOG1P = False
STATE_TARGET_MODE = "residual"
STATE_TARGET_SCALE_MODE = "state"
STATE_LOSS_NAME = "huber"
STATE_SAMPLE_WEIGHT_MODE = "inverse_sqrt_state_target_scale"

#pass through to gru_benchmark.py
def run_state_consistency_gru_benchmark(
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
        hidden_size=STATE_HIDDEN_SIZE,
        dropout=STATE_DROPOUT,
        batch_size=STATE_BATCH_SIZE,
        epochs=STATE_EPOCHS,
        learning_rate=STATE_LEARNING_RATE,
        weight_decay=STATE_WEIGHT_DECAY,
        use_log1p=STATE_USE_LOG1P,
        target_mode=STATE_TARGET_MODE,
        target_scale_mode=STATE_TARGET_SCALE_MODE,
        loss_name=STATE_LOSS_NAME,
        sample_weight_mode=STATE_SAMPLE_WEIGHT_MODE,
    )
