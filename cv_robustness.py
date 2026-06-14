import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from config import CV_N_SPLITS, RANDOM_STATE
from metrics import METRIC_NAMES, calc_ordinal_metrics
from models import make_model

# N_JOBS means to tell joblib.Parallel that how many CPU cores should be used for this project.
N_JOBS = 32


def _run_one_fold(fold, train_idx, valid_idx, X, y, model_name, params):
    """Single CV fold — runs in parallel."""
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    model = make_model(model_name, params)
    model.fit(X_train, y_train)

    pred = model.predict(X_valid)
    try:
        proba = model.predict_proba(X_valid)
    except Exception:
        proba = None

    metrics = calc_ordinal_metrics(
        y_valid, pred, y_proba=proba, classes_=np.array([0, 1, 2])
    )
    row = {"Model": model_name, "Fold": fold}
    row.update(metrics)
    return row


def cross_validation_robustness(
    model_name, params, X, y,
    n_splits=CV_N_SPLITS, random_state=RANDOM_STATE, n_jobs=N_JOBS,
):
    """
    Parallel K-fold CV robustness.
    All folds run simultaneously across CPU cores.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = Parallel(n_jobs=min(n_jobs, n_splits), backend="loky")(
        delayed(_run_one_fold)(fold, train_idx, valid_idx, X, y, model_name, params)
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1)
    )

    fold_df = pd.DataFrame(fold_rows)

    summary = {"Model": model_name}
    for m in METRIC_NAMES:
        summary[f"{m}_mean"] = fold_df[m].mean()
        summary[f"{m}_sd"]   = fold_df[m].std(ddof=1)

    return pd.DataFrame([summary]), fold_df
