from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_RNN_HIDDEN_SIZE = 96
DEFAULT_RNN_DROPOUT = 0.10
DEFAULT_RNN_BATCH_SIZE = 512
DEFAULT_RNN_EPOCHS = 10
DEFAULT_RNN_LEARNING_RATE = 8e-4
DEFAULT_RNN_WEIGHT_DECAY = 1e-5
DEFAULT_RNN_USE_LOG1P = False
DEFAULT_RNN_TARGET_MODE = "residual"
DEFAULT_RNN_LOSS_NAME = "huber"


def fit_sequence_preprocessor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    baseline_last_observed: np.ndarray | None = None,
    *,
    use_log1p: bool = DEFAULT_RNN_USE_LOG1P,
    target_mode: str = DEFAULT_RNN_TARGET_MODE,
) -> dict[str, object]:
    transformed_train_X = train_X.astype(np.float32).copy()
    transformed_train_y = train_y.astype(np.float32).copy()

    if use_log1p:
        transformed_train_X = np.log1p(np.clip(transformed_train_X, a_min=0.0, a_max=None))

    if target_mode == "raw":
        if use_log1p:
            transformed_train_y = np.log1p(np.clip(transformed_train_y, a_min=0.0, a_max=None))
    elif target_mode == "residual":
        if baseline_last_observed is None:
            raise ValueError("baseline_last_observed is required when target_mode='residual'.")
        transformed_train_y = transformed_train_y - baseline_last_observed.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    feature_mean = transformed_train_X.mean(axis=(0, 1), keepdims=True)
    feature_std = transformed_train_X.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    target_mean = float(transformed_train_y.mean())
    target_std = float(transformed_train_y.std())
    if target_std < 1e-6:
        target_std = 1.0

    return {
        "use_log1p": use_log1p,
        "target_mode": target_mode,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
    }


def transform_sequence_features(X: np.ndarray, preprocessor: dict[str, object]) -> np.ndarray:
    transformed = X.astype(np.float32).copy()

    if preprocessor["use_log1p"]:
        transformed = np.log1p(np.clip(transformed, a_min=0.0, a_max=None))

    transformed = (transformed - preprocessor["feature_mean"]) / preprocessor["feature_std"]
    return transformed.astype(np.float32)


def transform_sequence_targets(
    y: np.ndarray,
    preprocessor: dict[str, object],
    baseline_last_observed: np.ndarray | None = None,
) -> np.ndarray:
    transformed = y.astype(np.float32).copy()

    if preprocessor["target_mode"] == "raw":
        if preprocessor["use_log1p"]:
            transformed = np.log1p(np.clip(transformed, a_min=0.0, a_max=None))
    elif preprocessor["target_mode"] == "residual":
        if baseline_last_observed is None:
            raise ValueError("baseline_last_observed is required when target_mode='residual'.")
        transformed = transformed - baseline_last_observed.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target_mode: {preprocessor['target_mode']}")

    transformed = (transformed - preprocessor["target_mean"]) / preprocessor["target_std"]
    return transformed.astype(np.float32)


def inverse_transform_sequence_targets(
    y: np.ndarray,
    preprocessor: dict[str, object],
    baseline_last_observed: np.ndarray | None = None,
) -> np.ndarray:
    restored = y.astype(np.float32).copy()
    restored = restored * preprocessor["target_std"] + preprocessor["target_mean"]

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


def pick_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


class ForecastGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden_state = self.gru(x)
        last_hidden = self.dropout(hidden_state[-1])
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
) -> tuple[ForecastGRU, pd.DataFrame]:
    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    eval_dataset = TensorDataset(torch.from_numpy(eval_X), torch.from_numpy(eval_y))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = ForecastGRU(input_size=train_X.shape[2], hidden_size=hidden_size, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = get_loss_fn(loss_name)

    history_rows: list[dict[str, float | int | str]] = []
    best_eval_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch_index in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_row_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            batch_prediction = model(batch_X)
            batch_loss = loss_fn(batch_prediction, batch_y)
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
            for batch_X, batch_y in eval_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                batch_prediction = model(batch_X)
                batch_loss = loss_fn(batch_prediction, batch_y)

                batch_size_actual = batch_y.shape[0]
                eval_loss_sum += batch_loss.item() * batch_size_actual
                eval_row_count += batch_size_actual

        train_loss = train_loss_sum / train_row_count
        eval_loss = eval_loss_sum / eval_row_count
        history_rows.append(
            {
                "epoch": epoch_index + 1,
                "loss_name": loss_name,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }
        )

        if best_eval_loss is None or eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
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
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    prediction_rows: list[np.ndarray] = []

    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            batch_prediction = model(batch_X).detach().cpu().numpy()
            prediction_rows.append(batch_prediction)

    predictions = np.concatenate(prediction_rows, axis=0)
    return predictions.astype(np.float32)
