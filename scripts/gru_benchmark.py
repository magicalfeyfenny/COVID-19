from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

#define the actual network and training loop

DEFAULT_RNN_HIDDEN_SIZE = 96
DEFAULT_RNN_DROPOUT = 0.10
DEFAULT_RNN_BATCH_SIZE = 512
DEFAULT_RNN_EPOCHS = 50
DEFAULT_RNN_LEARNING_RATE = 8e-4
DEFAULT_RNN_WEIGHT_DECAY = 1e-5
DEFAULT_RNN_USE_LOG1P = False
DEFAULT_RNN_TARGET_MODE = "residual"
DEFAULT_RNN_TARGET_SCALE_MODE = "global"
DEFAULT_RNN_LOSS_NAME = "huber"
DEFAULT_RNN_SAMPLE_WEIGHT_MODE = "none"
DEFAULT_RNN_STATE_EMBEDDING_DIM = 0
DEFAULT_RNN_CHECKPOINT_METRIC_NAME = "eval_loss"


def build_base_sequence_targets(
    y: np.ndarray,
    *,
    use_log1p: bool,
    target_mode: str,
    baseline_last_observed: np.ndarray | None = None,
) -> np.ndarray:
    transformed = y.astype(np.float32).copy()

    if target_mode == "raw":
        if use_log1p:
            transformed = np.log1p(np.clip(transformed, a_min=0.0, a_max=None))
    elif target_mode == "residual":
        if baseline_last_observed is None:
            raise ValueError("baseline_last_observed is required when target_mode='residual'.")
        transformed = transformed - baseline_last_observed.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    return transformed.astype(np.float32)


def fit_sequence_preprocessor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_meta: pd.DataFrame | None = None,
    baseline_last_observed: np.ndarray | None = None,
    *,
    use_log1p: bool = DEFAULT_RNN_USE_LOG1P,
    target_mode: str = DEFAULT_RNN_TARGET_MODE,
    target_scale_mode: str = DEFAULT_RNN_TARGET_SCALE_MODE,
) -> dict[str, object]:
    transformed_train_X = train_X.astype(np.float32).copy()

    if use_log1p:
        transformed_train_X = np.log1p(np.clip(transformed_train_X, a_min=0.0, a_max=None))

    transformed_train_y = build_base_sequence_targets(
        train_y,
        use_log1p=use_log1p,
        target_mode=target_mode,
        baseline_last_observed=baseline_last_observed,
    )

    feature_mean = transformed_train_X.mean(axis=(0, 1), keepdims=True)
    feature_std = transformed_train_X.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    global_target_mean = float(transformed_train_y.mean())
    global_target_std = float(transformed_train_y.std())
    if global_target_std < 1e-6:
        global_target_std = 1.0

    state_target_mean: dict[str, float] = {}
    state_target_std: dict[str, float] = {}
    state_target_scale: dict[str, float] = {}
    state_to_index: dict[str, int] = {}

    if train_meta is not None:
        unique_states = sorted(train_meta["state"].astype(str).unique())
        state_to_index = {state_name: index for index, state_name in enumerate(unique_states)}

    if target_scale_mode == "state":
        if train_meta is None:
            raise ValueError("train_meta is required when target_scale_mode='state'.")

        state_frame = train_meta[["state"]].copy()
        state_frame["transformed_target"] = transformed_train_y
        state_frame["raw_target"] = train_y.astype(np.float32)

        grouped = state_frame.groupby("state")
        mean_series = grouped["transformed_target"].mean()
        std_series = grouped["transformed_target"].std().replace(0.0, np.nan).fillna(global_target_std)
        scale_series = grouped["raw_target"].mean().clip(lower=1.0)

        state_target_mean = mean_series.to_dict()
        state_target_std = std_series.to_dict()
        state_target_scale = scale_series.to_dict()
    elif target_scale_mode != "global":
        raise ValueError(f"Unsupported target_scale_mode: {target_scale_mode}")

    return {
        "use_log1p": use_log1p,
        "target_mode": target_mode,
        "target_scale_mode": target_scale_mode,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "global_target_mean": global_target_mean,
        "global_target_std": global_target_std,
        "state_target_mean": state_target_mean,
        "state_target_std": state_target_std,
        "state_target_scale": state_target_scale,
        "state_to_index": state_to_index,
    }


def encode_state_indices(meta_frame: pd.DataFrame | None, preprocessor: dict[str, object]) -> np.ndarray:
    if meta_frame is None:
        raise ValueError("meta_frame is required to encode state indices.")

    state_to_index = dict(preprocessor.get("state_to_index", {}))
    if not state_to_index:
        return np.zeros(len(meta_frame), dtype=np.int64)

    encoded = meta_frame["state"].map(state_to_index)
    if encoded.isna().any():
        missing_states = sorted(meta_frame.loc[encoded.isna(), "state"].astype(str).unique())
        raise ValueError(f"Found states without an index mapping: {missing_states}")

    return encoded.to_numpy(dtype=np.int64)


def transform_sequence_features(X: np.ndarray, preprocessor: dict[str, object]) -> np.ndarray:
    transformed = X.astype(np.float32).copy()

    if preprocessor["use_log1p"]:
        transformed = np.log1p(np.clip(transformed, a_min=0.0, a_max=None))

    transformed = (transformed - preprocessor["feature_mean"]) / preprocessor["feature_std"]
    return transformed.astype(np.float32)


def transform_sequence_targets(
    y: np.ndarray,
    preprocessor: dict[str, object],
    meta_frame: pd.DataFrame | None = None,
    baseline_last_observed: np.ndarray | None = None,
) -> np.ndarray:
    transformed = build_base_sequence_targets(
        y,
        use_log1p=bool(preprocessor["use_log1p"]),
        target_mode=str(preprocessor["target_mode"]),
        baseline_last_observed=baseline_last_observed,
    )

    if preprocessor["target_scale_mode"] == "state":
        if meta_frame is None:
            raise ValueError("meta_frame is required when target_scale_mode='state'.")

        state_mean = meta_frame["state"].map(preprocessor["state_target_mean"]).to_numpy(dtype=np.float32)
        state_std = meta_frame["state"].map(preprocessor["state_target_std"]).to_numpy(dtype=np.float32)
        transformed = (transformed - state_mean) / state_std
    else:
        transformed = (transformed - preprocessor["global_target_mean"]) / preprocessor["global_target_std"]

    return transformed.astype(np.float32)


def inverse_transform_sequence_targets(
    y: np.ndarray,
    preprocessor: dict[str, object],
    meta_frame: pd.DataFrame | None = None,
    baseline_last_observed: np.ndarray | None = None,
) -> np.ndarray:
    restored = y.astype(np.float32).copy()

    if preprocessor["target_scale_mode"] == "state":
        if meta_frame is None:
            raise ValueError("meta_frame is required when target_scale_mode='state'.")

        state_mean = meta_frame["state"].map(preprocessor["state_target_mean"]).to_numpy(dtype=np.float32)
        state_std = meta_frame["state"].map(preprocessor["state_target_std"]).to_numpy(dtype=np.float32)
        restored = restored * state_std + state_mean
    else:
        restored = restored * preprocessor["global_target_std"] + preprocessor["global_target_mean"]

    if preprocessor["target_mode"] == "raw":
        if preprocessor["use_log1p"]:
            restored = np.expm1(restored)
    elif preprocessor["target_mode"] == "residual":
        if baseline_last_observed is None:
            raise ValueError("baseline_last_observed is required when target_mode='residual'.")
        restored = restored + baseline_last_observed.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target_mode: {preprocessor['target_mode']}")

    restored = np.clip(restored, a_min=0.0, a_max=None)
    return restored.astype(np.float32)


def build_sequence_sample_weights(
    meta_frame: pd.DataFrame | None,
    preprocessor: dict[str, object],
    *,
    sample_weight_mode: str = DEFAULT_RNN_SAMPLE_WEIGHT_MODE,
) -> np.ndarray | None:
    if sample_weight_mode == "none":
        return None

    if sample_weight_mode == "inverse_sqrt_state_target_scale":
        if meta_frame is None:
            raise ValueError("meta_frame is required when sample_weight_mode is state-based.")

        state_scale = meta_frame["state"].map(preprocessor["state_target_scale"]).to_numpy(dtype=np.float32)
        weights = 1.0 / np.sqrt(np.clip(state_scale, a_min=1.0, a_max=None))
        weights = weights / weights.mean()
        return weights.astype(np.float32)

    if sample_weight_mode == "equal_state_total":
        if meta_frame is None:
            raise ValueError("meta_frame is required when sample_weight_mode is state-based.")

        state_counts = meta_frame["state"].value_counts().astype(np.float32)
        state_weights = 1.0 / state_counts
        weights = meta_frame["state"].map(state_weights).to_numpy(dtype=np.float32)
        weights = weights / weights.mean()
        return weights.astype(np.float32)

    raise ValueError(f"Unsupported sample_weight_mode: {sample_weight_mode}")


def compute_mean_state_mae(actual: np.ndarray, predicted: np.ndarray, meta_frame: pd.DataFrame) -> float:
    score_frame = meta_frame[["state"]].copy()
    score_frame["actual"] = actual.astype(np.float32)
    score_frame["predicted"] = predicted.astype(np.float32)
    score_frame["abs_error"] = (score_frame["actual"] - score_frame["predicted"]).abs()

    per_state_mae = score_frame.groupby("state", sort=True)["abs_error"].mean()
    return float(per_state_mae.mean())


def pick_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loss_fn(loss_name: str, *, reduction: str = "mean") -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss(reduction=reduction)
    if loss_name == "mae":
        return nn.L1Loss(reduction=reduction)
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=1.0, reduction=reduction)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


class ForecastGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        *,
        state_vocab_size: int = 0,
        state_embedding_dim: int = DEFAULT_RNN_STATE_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_state_embedding = state_vocab_size > 0 and state_embedding_dim > 0

        if self.use_state_embedding:
            self.state_embedding = nn.Embedding(state_vocab_size, state_embedding_dim)
            output_input_size = hidden_size + state_embedding_dim
        else:
            self.state_embedding = None
            output_input_size = hidden_size

        self.output = nn.Linear(output_input_size, 1)

    def forward(self, x: torch.Tensor, state_index: torch.Tensor | None = None) -> torch.Tensor:
        _, hidden_state = self.gru(x)
        last_hidden = self.dropout(hidden_state[-1])

        if self.use_state_embedding:
            if state_index is None:
                raise ValueError("state_index is required when state embeddings are enabled.")
            state_hidden = self.state_embedding(state_index)
            last_hidden = torch.cat([last_hidden, state_hidden], dim=1)

        prediction = self.output(last_hidden).squeeze(-1)
        return prediction


def train_gru_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    device: torch.device,
    *,
    hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE,
    dropout: float = DEFAULT_RNN_DROPOUT,
    batch_size: int = DEFAULT_RNN_BATCH_SIZE,
    epochs: int = DEFAULT_RNN_EPOCHS,
    learning_rate: float = DEFAULT_RNN_LEARNING_RATE,
    weight_decay: float = DEFAULT_RNN_WEIGHT_DECAY,
    loss_name: str = DEFAULT_RNN_LOSS_NAME,
    train_sample_weights: np.ndarray | None = None,
    eval_sample_weights: np.ndarray | None = None,
    train_state_index: np.ndarray | None = None,
    eval_state_index: np.ndarray | None = None,
    state_vocab_size: int = 0,
    state_embedding_dim: int = DEFAULT_RNN_STATE_EMBEDDING_DIM,
    checkpoint_metric_name: str = DEFAULT_RNN_CHECKPOINT_METRIC_NAME,
    checkpoint_metric_fn: Callable[[ForecastGRU], float] | None = None,
) -> tuple[ForecastGRU, pd.DataFrame]:
    if train_sample_weights is None:
        train_sample_weights = np.ones(len(train_y), dtype=np.float32)
    if eval_sample_weights is None:
        eval_sample_weights = np.ones(len(eval_y), dtype=np.float32)
    if train_state_index is None:
        train_state_index = np.zeros(len(train_y), dtype=np.int64)
    if eval_state_index is None:
        eval_state_index = np.zeros(len(eval_y), dtype=np.int64)

    if checkpoint_metric_name != "eval_loss" and checkpoint_metric_fn is None:
        raise ValueError("checkpoint_metric_fn is required when checkpoint_metric_name is not 'eval_loss'.")

    train_dataset = TensorDataset(
        torch.from_numpy(train_X),
        torch.from_numpy(train_y),
        torch.from_numpy(train_sample_weights.astype(np.float32)),
        torch.from_numpy(train_state_index.astype(np.int64)),
    )
    eval_dataset = TensorDataset(
        torch.from_numpy(eval_X),
        torch.from_numpy(eval_y),
        torch.from_numpy(eval_sample_weights.astype(np.float32)),
        torch.from_numpy(eval_state_index.astype(np.int64)),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = ForecastGRU(
        input_size=train_X.shape[2],
        hidden_size=hidden_size,
        dropout=dropout,
        state_vocab_size=state_vocab_size,
        state_embedding_dim=state_embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = get_loss_fn(loss_name, reduction="none")

    history_rows: list[dict[str, float | int | str]] = []
    best_checkpoint_metric: float | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch_index in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_row_count = 0

        for batch_X, batch_y, batch_weights, batch_state_index in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_weights = batch_weights.to(device)
            batch_state_index = batch_state_index.to(device)

            optimizer.zero_grad()
            batch_prediction = model(batch_X, batch_state_index)
            batch_loss = loss_fn(batch_prediction, batch_y)
            batch_loss = (batch_loss * batch_weights).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = batch_y.shape[0]
            train_loss_sum += batch_loss.item() * batch_size_actual
            train_row_count += batch_size_actual

        model.eval()
        eval_loss_sum = 0.0
        eval_row_count = 0

        with torch.no_grad():
            for batch_X, batch_y, batch_weights, batch_state_index in eval_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_weights = batch_weights.to(device)
                batch_state_index = batch_state_index.to(device)

                batch_prediction = model(batch_X, batch_state_index)
                batch_loss = loss_fn(batch_prediction, batch_y)
                batch_loss = (batch_loss * batch_weights).mean()

                batch_size_actual = batch_y.shape[0]
                eval_loss_sum += batch_loss.item() * batch_size_actual
                eval_row_count += batch_size_actual

        train_loss = train_loss_sum / train_row_count
        eval_loss = eval_loss_sum / eval_row_count

        if checkpoint_metric_name == "eval_loss":
            checkpoint_metric_value = eval_loss
        else:
            checkpoint_metric_value = float(checkpoint_metric_fn(model))

        history_rows.append(
            {
                "epoch": epoch_index + 1,
                "loss_name": loss_name,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "checkpoint_metric_name": checkpoint_metric_name,
                "checkpoint_metric_value": checkpoint_metric_value,
            }
        )

        if best_checkpoint_metric is None or checkpoint_metric_value < best_checkpoint_metric:
            best_checkpoint_metric = checkpoint_metric_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    return model, history


def predict_gru_model(
    model: ForecastGRU,
    features: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = DEFAULT_RNN_BATCH_SIZE,
    state_index: np.ndarray | None = None,
) -> np.ndarray:
    if state_index is None:
        state_index = np.zeros(len(features), dtype=np.int64)

    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(state_index.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    prediction_rows: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, batch_state_index in loader:
            batch_X = batch_X.to(device)
            batch_state_index = batch_state_index.to(device)
            batch_prediction = model(batch_X, batch_state_index).detach().cpu().numpy()
            prediction_rows.append(batch_prediction)

    predictions = np.concatenate(prediction_rows, axis=0)
    return predictions.astype(np.float32)


def run_gru_benchmark(
    *,
    model_key: str,
    display_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_meta: pd.DataFrame,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    eval_meta: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    random_seed: int,
    hidden_size: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    use_log1p: bool,
    target_mode: str,
    target_scale_mode: str,
    loss_name: str,
    sample_weight_mode: str,
    state_embedding_dim: int = DEFAULT_RNN_STATE_EMBEDDING_DIM,
    checkpoint_metric_name: str = DEFAULT_RNN_CHECKPOINT_METRIC_NAME,
) -> dict[str, object]:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    target_feature_index = list(feature_columns).index(target_column)
    train_last_observed = train_X[:, -1, target_feature_index]
    eval_last_observed = eval_X[:, -1, target_feature_index]

    preprocessor = fit_sequence_preprocessor(
        train_X,
        train_y,
        train_meta=train_meta,
        baseline_last_observed=train_last_observed,
        use_log1p=use_log1p,
        target_mode=target_mode,
        target_scale_mode=target_scale_mode,
    )

    train_X_scaled = transform_sequence_features(train_X, preprocessor)
    eval_X_scaled = transform_sequence_features(eval_X, preprocessor)
    train_state_index = encode_state_indices(train_meta, preprocessor)
    eval_state_index = encode_state_indices(eval_meta, preprocessor)
    train_y_scaled = transform_sequence_targets(
        train_y,
        preprocessor,
        meta_frame=train_meta,
        baseline_last_observed=train_last_observed,
    )
    eval_y_scaled = transform_sequence_targets(
        eval_y,
        preprocessor,
        meta_frame=eval_meta,
        baseline_last_observed=eval_last_observed,
    )

    train_sample_weights = build_sequence_sample_weights(
        train_meta,
        preprocessor,
        sample_weight_mode=sample_weight_mode,
    )
    eval_sample_weights = build_sequence_sample_weights(
        eval_meta,
        preprocessor,
        sample_weight_mode=sample_weight_mode,
    )

    device = pick_torch_device()

    checkpoint_metric_fn: Callable[[ForecastGRU], float] | None = None
    if checkpoint_metric_name == "mean_state_mae":
        def checkpoint_metric_fn(model: ForecastGRU) -> float:
            prediction_scaled = predict_gru_model(
                model,
                eval_X_scaled,
                device=device,
                batch_size=batch_size,
                state_index=eval_state_index,
            )
            predictions = inverse_transform_sequence_targets(
                prediction_scaled,
                preprocessor,
                meta_frame=eval_meta,
                baseline_last_observed=eval_last_observed,
            )
            return compute_mean_state_mae(eval_y, predictions, eval_meta)
    elif checkpoint_metric_name != "eval_loss":
        raise ValueError(f"Unsupported checkpoint_metric_name: {checkpoint_metric_name}")

    model, history = train_gru_model(
        train_X_scaled,
        train_y_scaled,
        eval_X_scaled,
        eval_y_scaled,
        device=device,
        hidden_size=hidden_size,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_name=loss_name,
        train_sample_weights=train_sample_weights,
        eval_sample_weights=eval_sample_weights,
        train_state_index=train_state_index,
        eval_state_index=eval_state_index,
        state_vocab_size=len(dict(preprocessor.get("state_to_index", {}))),
        state_embedding_dim=state_embedding_dim,
        checkpoint_metric_name=checkpoint_metric_name,
        checkpoint_metric_fn=checkpoint_metric_fn,
    )

    prediction_column = f"{model_key}_prediction"
    prediction_scaled = predict_gru_model(
        model,
        eval_X_scaled,
        device=device,
        batch_size=batch_size,
        state_index=eval_state_index,
    )
    predictions = inverse_transform_sequence_targets(
        prediction_scaled,
        preprocessor,
        meta_frame=eval_meta,
        baseline_last_observed=eval_last_observed,
    )

    history = history.copy()
    history.insert(0, "model_key", model_key)
    history.insert(1, "display_name", display_name)
    history["device"] = str(device)

    return {
        "model_key": model_key,
        "display_name": display_name,
        "prediction_column": prediction_column,
        "predictions": predictions,
        "history": history,
        "device": str(device),
        "preprocessor": preprocessor,
    }
