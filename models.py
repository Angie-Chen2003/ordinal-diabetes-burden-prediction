import random

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


def set_all_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)


class XGBOrdinalModel:
    """
    Wrapper for xgbordinal.XGBOrdinal.

    If xgbordinal is unavailable, install:
      python -m pip install xgbordinal xgboost
    """
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        from xgbordinal import XGBOrdinal

        params = self.params.copy()

        # Accept both n_estimator and n_estimators.
        if "n_estimator" in params and "n_estimators" not in params:
            params["n_estimators"] = params.pop("n_estimator")

        self.classes_ = np.sort(np.unique(y))
        self.model = XGBOrdinal(**params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).astype(int)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X)
            except Exception:
                return None
        return None


class CoralTorchModel:
    """
    Simple CORAL-style ordinal neural network implemented with PyTorch.

    The model learns K-1 binary cumulative logits:
      P(Y > 0), P(Y > 1)

    For 3 classes, predicted class = number of thresholds passed.

    Install if needed:
      python -m pip install torch
    """
    def __init__(
        self,
        hidden1=32,
        hidden2=0,
        dropout=0.2,
        lr=0.003,
        weight_decay=1e-4,
        batch_size=64,
        epochs=150,
        random_state=123,
        verbose=False,
    ):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.model = None
        self.classes_ = np.array([0, 1, 2])

    def _build_model(self, input_dim, n_thresholds):
        import torch
        import torch.nn as nn

        layers = []
        layers.append(nn.Linear(input_dim, int(self.hidden1)))
        layers.append(nn.ReLU())
        if self.dropout and self.dropout > 0:
            layers.append(nn.Dropout(float(self.dropout)))

        if self.hidden2 and int(self.hidden2) > 0:
            layers.append(nn.Linear(int(self.hidden1), int(self.hidden2)))
            layers.append(nn.ReLU())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            layers.append(nn.Linear(int(self.hidden2), n_thresholds))
        else:
            layers.append(nn.Linear(int(self.hidden1), n_thresholds))

        return nn.Sequential(*layers)

    @staticmethod
    def _y_to_cumulative(y, n_classes):
        """
        For classes 0, 1, 2:
          y=0 -> [0, 0]
          y=1 -> [1, 0]
          y=2 -> [1, 1]
        """
        y = np.asarray(y).astype(int)
        out = np.zeros((len(y), n_classes - 1), dtype=np.float32)
        for k in range(n_classes - 1):
            out[:, k] = (y > k).astype(np.float32)
        return out

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        set_all_seeds(self.random_state)
        torch.manual_seed(self.random_state)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=int)

        self.classes_ = np.sort(np.unique(y_np))
        n_classes = int(len(self.classes_))
        if n_classes < 2:
            raise ValueError("CORAL requires at least two classes.")

        # This project expects classes 0,1,2. If a split has missing class,
        # the output still uses full 3-class structure.
        n_classes = max(3, n_classes)
        n_thresholds = n_classes - 1

        X_scaled = self.scaler.fit_transform(X_np).astype(np.float32)
        y_cum = self._y_to_cumulative(y_np, n_classes=n_classes)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_cum, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            drop_last=False,
        )

        self.model = self._build_model(
            input_dim=X_scaled.shape[1],
            n_thresholds=n_thresholds,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(int(self.epochs)):
            losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"CORAL epoch {epoch + 1}/{self.epochs}, loss={np.mean(losses):.4f}")

        return self

    def predict_proba(self, X):
        import torch

        X_np = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_np).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            cum_probs = torch.sigmoid(logits).cpu().numpy()

        # Enforce monotonicity approximately: P(Y>0) >= P(Y>1)
        if cum_probs.shape[1] >= 2:
            cum_probs[:, 1] = np.minimum(cum_probs[:, 0], cum_probs[:, 1])

        # For 3 classes:
        # P(Y=0)=1-P(Y>0)
        # P(Y=1)=P(Y>0)-P(Y>1)
        # P(Y=2)=P(Y>1)
        p0 = 1 - cum_probs[:, 0]
        p1 = cum_probs[:, 0] - cum_probs[:, 1]
        p2 = cum_probs[:, 1]
        proba = np.vstack([p0, p1, p2]).T
        proba = np.clip(proba, 0, 1)
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        proba = proba / row_sum
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(int)


def make_model(model_name, params):
    model_name = str(model_name).upper()
    params = params.copy()

    if model_name in {"XGB", "XGBOOST", "ORDINAL XGBOOST", "XGBORDINAL"}:
        return XGBOrdinalModel(**params)

    if model_name in {"CORAL", "CORALTORCH"}:
        return CoralTorchModel(**params)

    raise ValueError(f"Unknown model_name: {model_name}")
