"""Optional PyTorch sequence model for multivariate IV-surface forecasts."""
from __future__ import annotations

import numpy as np


def fit_predict_iv_lstm(matrix: np.ndarray, sequence_length: int = 5,
                        seed: int = 42, max_epochs: int = 60,
                        hidden_size: int = 32) -> np.ndarray:
    """Predict the next complete IV grid from prior surface snapshots.

    ``matrix`` must contain training dates only and be imputed using training
    data only. Each input is a sequence of complete surfaces; each target is
    the next surface. Returns NaNs when there are too few training sequences.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("IV LSTM matrix must be a finite date-by-cell array")
    if sequence_length < 2 or max_epochs < 1 or hidden_size < 1:
        raise ValueError("sequence_length must be >= 2; epochs and hidden size must be positive")
    if len(values) - sequence_length < 8:
        return np.full(values.shape[1], np.nan)
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("Install the optional PyTorch dependency with `pip install -e '.[ml-deep]'`") from exc

    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-7] = 1.0
    normalized = (values - center) / scale
    windows = np.stack([normalized[i:i + sequence_length]
                        for i in range(len(values) - sequence_length)])
    targets = normalized[sequence_length:]
    split = max(1, int(len(windows) * 0.85))
    if len(windows) - split < 1:
        split = len(windows) - 1
    # Python-list conversion also works with PyTorch 2.2.x on Intel macOS,
    # whose wheel cannot use the NumPy 2.x ndarray bridge.
    x_fit = torch.tensor(windows[:split].tolist(), dtype=torch.float32)
    y_fit = torch.tensor(targets[:split].tolist(), dtype=torch.float32)
    x_valid = torch.tensor(windows[split:].tolist(), dtype=torch.float32)
    y_valid = torch.tensor(targets[split:].tolist(), dtype=torch.float32)
    latest = torch.tensor(normalized[-sequence_length:][None].tolist(), dtype=torch.float32)
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

        class SurfaceLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(values.shape[1], hidden_size, batch_first=True)
                self.dropout = nn.Dropout(0.1)
                self.output = nn.Linear(hidden_size, values.shape[1])

            def forward(self, surface_sequence):
                sequence, _ = self.lstm(surface_sequence)
                return self.output(self.dropout(sequence[:, -1, :]))

        model = SurfaceLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_fit), y_fit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(model(x_valid), y_valid))
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_state = {key: value.detach().clone()
                              for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= 10:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = np.asarray(model(latest).squeeze(0).tolist(), dtype=float)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        torch.set_num_threads(prior_threads)
    return np.maximum(center + prediction * scale, 0.0)
