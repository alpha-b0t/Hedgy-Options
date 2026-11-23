"""Optional PyTorch LSTM for rolling-window RV forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.models.ml import ML_FEATURES


def lstm_available() -> bool:
    """Return whether PyTorch can be imported in the current environment."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def fit_predict_lstm(data: pd.DataFrame, target: str, train_limit: int,
                     forecast_indices: np.ndarray, sequence_length: int = 20,
                     seed: int = 42, max_epochs: int = 60) -> np.ndarray:
    """Fit on labeled target rows before ``train_limit`` and forecast windows.

    Features from earlier rows are allowed as sequence context. Labels at or
    after ``train_limit`` are never used. Scaling and the chronological
    early-stopping split are fit solely on the training fold.
    """
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("Install the optional LSTM dependency with `pip install -e '.[ml-deep]'`") from exc
    if sequence_length < 2 or max_epochs < 1:
        raise ValueError("sequence_length must be >= 2 and max_epochs must be positive")
    if target not in data or set(ML_FEATURES) - set(data.columns):
        raise ValueError("LSTM input is missing its target or required ML features")
    frame = data.reset_index(drop=True)
    x = frame.loc[:, ML_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    indices = np.asarray(forecast_indices, dtype=int)
    if not len(indices) or np.any(indices < sequence_length - 1) or np.any(indices >= len(frame)):
        return np.full(len(indices), np.nan, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(x, sequence_length, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    target_rows = np.arange(sequence_length - 1, len(frame))
    valid_window = np.isfinite(windows).all(axis=(1, 2))
    train_mask = (target_rows < int(train_limit)) & np.isfinite(y[target_rows]) & valid_window
    if train_mask.sum() < 50:
        return np.full(len(indices), np.nan, dtype=float)
    forecast_positions = indices - sequence_length + 1
    forecast_mask = valid_window[forecast_positions]
    if not forecast_mask.any():
        return np.full(len(indices), np.nan, dtype=float)

    train_x = windows[train_mask].astype(np.float32)
    train_y = y[target_rows[train_mask]].astype(np.float32)
    feature_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    feature_scale = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    train_x = (train_x - feature_mean) / feature_scale
    y_mean, y_scale = float(train_y.mean()), float(train_y.std())
    if y_scale < 1e-8:
        y_scale = 1.0
    train_y = (train_y - y_mean) / y_scale

    split = max(1, int(len(train_x) * 0.85))
    if len(train_x) - split < 1:
        split = len(train_x) - 1
    x_fit = torch.from_numpy(train_x[:split])
    y_fit = torch.from_numpy(train_y[:split, None])
    x_valid = torch.from_numpy(train_x[split:])
    y_valid = torch.from_numpy(train_y[split:, None])
    x_forecast = windows[forecast_positions[forecast_mask]].astype(np.float32)
    x_forecast = torch.from_numpy((x_forecast - feature_mean) / feature_scale)

    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        model = _LSTMRegressor(input_size=len(ML_FEATURES), hidden_size=16, nn_module=nn)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        loss_function = nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        patience = 9
        stale_epochs = 0
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_fit), y_fit)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = (float(loss_function(model(x_valid), y_valid))
                                   if len(x_valid) else float(loss))
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions = model(x_forecast).squeeze(-1).numpy() * y_scale + y_mean
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        torch.set_num_threads(prior_threads)
    output = np.full(len(indices), np.nan, dtype=float)
    output[forecast_mask] = np.maximum(predictions, 0.0)
    return output


class _LSTMRegressor:
    """Thin module wrapper kept local so torch remains optional at import time."""

    def __new__(cls, input_size: int, hidden_size: int, nn_module):
        class Model(nn_module.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn_module.LSTM(input_size, hidden_size, batch_first=True)
                self.output = nn_module.Linear(hidden_size, 1)

            def forward(self, values):
                sequence, _ = self.lstm(values)
                return self.output(sequence[:, -1, :])

        return Model()
