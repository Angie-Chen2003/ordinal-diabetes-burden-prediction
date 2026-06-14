import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from metrics import METRIC_NAMES, calc_ordinal_metrics

# Number of parallel jobs — matches --cpus-per-task in SLURM
N_JOBS = 32


def percentile_ci(values, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    lower = np.percentile(arr, 100 * alpha / 2)
    upper = np.percentile(arr, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def format_ci(lower, upper, digits=3):
    if np.isnan(lower) or np.isnan(upper):
        return ""
    return f"({lower:.{digits}f}, {upper:.{digits}f})"


def _one_bootstrap_metric(b, y_true, y_pred, y_proba_arr, n, classes_, seed):
    """Single bootstrap iteration for metric CI — runs in parallel."""
    rng = np.random.default_rng(seed + b)
    idx = rng.integers(0, n, size=n)
    yp = y_pred[idx]
    yt = y_true[idx]
    pp = None if y_proba_arr is None else y_proba_arr[idx, :]
    try:
        return calc_ordinal_metrics(yt, yp, pp, classes_=classes_)
    except Exception:
        return {m: np.nan for m in METRIC_NAMES}


def bootstrap_metric_ci(
    y_true, y_pred, y_proba=None, classes_=None,
    B=1000, random_state=123, n_jobs=N_JOBS,
):
    """
    Parallel bootstrap CI for evaluation metrics.
    Runs B bootstrap iterations across n_jobs CPU cores simultaneously.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba_arr = None if y_proba is None else np.asarray(y_proba)
    n = len(y_true)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_bootstrap_metric)(
            b, y_true, y_pred, y_proba_arr, n, classes_, random_state
        )
        for b in range(B)
    )

    boot_values = {m: [r[m] for r in results] for m in METRIC_NAMES}

    ci = {}
    for m in METRIC_NAMES:
        lo, hi = percentile_ci(boot_values[m])
        ci[f"{m}_CI_Lower"] = lo
        ci[f"{m}_CI_Upper"] = hi
        ci[f"{m}_CI"] = format_ci(lo, hi)

    return ci


def add_metric_ci_to_row(row, ci_dict):
    out = row.copy()
    for m in METRIC_NAMES:
        out[f"{m}_CI"] = ci_dict.get(f"{m}_CI", "")
        out[f"{m}_CI_Lower"] = ci_dict.get(f"{m}_CI_Lower", np.nan)
        out[f"{m}_CI_Upper"] = ci_dict.get(f"{m}_CI_Upper", np.nan)
    return out


def _one_bootstrap_importance(
    b, fitted_model, X_np, y, predictors, metric_name, n_repeats, seed
):
    """Single bootstrap iteration for permutation importance — runs in parallel."""
    rng = np.random.default_rng(seed + b)
    n = len(y)
    idx = rng.integers(0, n, size=n)

    X_df = pd.DataFrame(X_np[idx], columns=predictors)
    yb = y[idx]
    classes_ = np.array([0, 1, 2])

    base_pred = fitted_model.predict(X_df)
    try:
        base_proba = fitted_model.predict_proba(X_df)
    except Exception:
        base_proba = None

    base_metrics = calc_ordinal_metrics(yb, base_pred, base_proba, classes_=classes_)
    base_value = base_metrics.get(metric_name, np.nan)

    perm_means = {}
    importances = {}
    for p in predictors:
        perm_vals = []
        for _ in range(n_repeats):
            Xp = X_df.copy()
            Xp[p] = rng.permutation(Xp[p].to_numpy())
            pred_p = fitted_model.predict(Xp)
            try:
                proba_p = fitted_model.predict_proba(Xp)
            except Exception:
                proba_p = None
            m_p = calc_ordinal_metrics(yb, pred_p, proba_p, classes_=classes_)
            perm_vals.append(m_p.get(metric_name, np.nan))

        perm_mean = float(np.nanmean(perm_vals))
        perm_means[p] = perm_mean
        importances[p] = base_value - perm_mean

    return base_value, perm_means, importances


def permutation_importance_with_bootstrap_ci(
    fitted_model, X, y,
    metric_name="ORC",
    B=1000, n_repeats=10,
    random_state=123, n_jobs=N_JOBS,
):
    """
    Parallel bootstrap uncertainty for permutation variable importance.
    Runs B bootstrap iterations across n_jobs CPU cores simultaneously.
    """
    X = X.copy()
    y = np.asarray(y)
    predictors = list(X.columns)
    X_np = X.to_numpy()

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_bootstrap_importance)(
            b, fitted_model, X_np, y, predictors,
            metric_name, n_repeats, random_state
        )
        for b in range(B)
    )

    all_base = [r[0] for r in results]
    all_perm = {p: [r[1][p] for r in results] for p in predictors}
    all_imp  = {p: [r[2][p] for r in results] for p in predictors}

    base_metric_mean = float(np.nanmean(all_base))
    rows = []
    for p in predictors:
        imp_arr  = np.asarray(all_imp[p],  dtype=float)
        perm_arr = np.asarray(all_perm[p], dtype=float)
        lo, hi = percentile_ci(imp_arr)
        rows.append({
            "Predictor": p,
            f"Base_{metric_name}": base_metric_mean,
            f"Permuted_{metric_name}_Mean": float(np.nanmean(perm_arr)),
            "Importance": float(np.nanmean(imp_arr)),
            "Importance_CI_Lower": lo,
            "Importance_CI_Upper": hi,
            "CI Importance": format_ci(lo, hi),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    return out
