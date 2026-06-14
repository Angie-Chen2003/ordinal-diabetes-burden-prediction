import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

# Here are the main indexes for this project.
METRIC_NAMES = [
    "Accuracy",
    "Macro_F1",
    "MAE",
    "MSE",
    "MZOE",
    "Macro_MAE",
    "ORC",
    "GC",
    "ADC",
]


def _as_int_array(y):
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        y = pd.Series(y).astype("category").cat.codes.to_numpy()
    return y.astype(int)


def _safe_auc_binary(y_binary, score):
    y_binary = np.asarray(y_binary).astype(int)
    score = np.asarray(score)
    if len(np.unique(y_binary)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_binary, score))
    except Exception:
        return np.nan


def _get_score_matrix(y_pred=None, y_proba=None, classes_=None):
    """
    Return a score matrix with shape n x K.
    Prefer model probabilities if available.
    If unavailable, use deterministic pseudo-probabilities from predicted class.
    """
    if y_proba is not None:
        arr = np.asarray(y_proba)
        if arr.ndim == 2:
            return arr

    y_pred = _as_int_array(y_pred)
    if classes_ is None:
        classes_ = np.sort(np.unique(y_pred))
    classes_ = np.asarray(classes_, dtype=int)
    out = np.zeros((len(y_pred), len(classes_)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes_)}
    for i, yp in enumerate(y_pred):
        if yp in class_to_idx:
            out[i, class_to_idx[yp]] = 1.0
    return out


def _expected_score_from_proba(y_pred=None, y_proba=None, classes_=None):
    if y_proba is None:
        return _as_int_array(y_pred).astype(float)

    proba = np.asarray(y_proba)
    if proba.ndim != 2:
        return _as_int_array(y_pred).astype(float)

    if classes_ is None:
        classes_ = np.arange(proba.shape[1])
    classes_ = np.asarray(classes_, dtype=float)
    return proba @ classes_


def ordinal_c_index(y_binary, score):
    """
    Binary c-index / AUC equivalent using roc_auc_score.
    y_binary = 1 for higher ordinal category/group.
    """
    return _safe_auc_binary(y_binary, score)


def calc_orc_gc_adc(y_true, y_pred, y_proba=None, classes_=None):
    """
    ORC:
      Average pairwise c-index across all ordered class pairs.

    GC:
      Pairwise c-index weighted by pair sample sizes.

    ADC:
      Average AUC across all cumulative splits P(Y > j).
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)

    if classes_ is None:
        classes_ = np.sort(np.unique(y_true))
    classes_ = np.asarray(classes_, dtype=int)

    expected_score = _expected_score_from_proba(
        y_pred=y_pred,
        y_proba=y_proba,
        classes_=classes_,
    )

    # ORC and GC
    pair_aucs = []
    pair_weights = []
    for i, low_class in enumerate(classes_):
        for high_class in classes_[i + 1:]:
            mask = np.isin(y_true, [low_class, high_class])
            if mask.sum() == 0:
                continue
            yy = (y_true[mask] == high_class).astype(int)
            ss = expected_score[mask]
            auc = _safe_auc_binary(yy, ss)
            if not np.isnan(auc):
                pair_aucs.append(auc)
                pair_weights.append(mask.sum())

    orc = float(np.nanmean(pair_aucs)) if len(pair_aucs) else np.nan
    gc = float(np.average(pair_aucs, weights=pair_weights)) if len(pair_aucs) else np.nan

    # ADC cumulative AUC: Y > j
    proba = _get_score_matrix(y_pred=y_pred, y_proba=y_proba, classes_=classes_)
    cumulative_aucs = []
    for j_idx in range(len(classes_) - 1):
        threshold_class = classes_[j_idx]
        y_bin = (y_true > threshold_class).astype(int)

        # Score = predicted probability of being above threshold.
        if proba.shape[1] == len(classes_):
            score = proba[:, j_idx + 1:].sum(axis=1)
        else:
            score = expected_score

        auc = _safe_auc_binary(y_bin, score)
        if not np.isnan(auc):
            cumulative_aucs.append(auc)

    adc = float(np.nanmean(cumulative_aucs)) if len(cumulative_aucs) else np.nan

    return orc, gc, adc


def macro_mae(y_true, y_pred):
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    values = []
    for cls in np.sort(np.unique(y_true)):
        mask = y_true == cls
        if mask.sum() > 0:
            values.append(mean_absolute_error(y_true[mask], y_pred[mask]))
    return float(np.mean(values)) if values else np.nan


def calc_ordinal_metrics(y_true, y_pred, y_proba=None, classes_=None):
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)

    if classes_ is None:
        classes_ = np.sort(np.unique(y_true))

    orc, gc, adc = calc_orc_gc_adc(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes_=classes_,
    )

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MZOE": float(np.mean(y_true != y_pred)),
        "Macro_MAE": macro_mae(y_true, y_pred),
        "ORC": orc,
        "GC": gc,
        "ADC": adc,
    }
    return out


def metrics_to_row(model_name, dataset_name, metrics_dict):
    row = {
        "Model": model_name,
        "Dataset": dataset_name,
    }
    row.update({m: metrics_dict.get(m, np.nan) for m in METRIC_NAMES})
    return row
