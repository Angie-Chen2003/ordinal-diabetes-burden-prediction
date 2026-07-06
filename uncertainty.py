import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from metrics import METRIC_NAMES, calc_ordinal_metrics

# Number of parallel jobs — matches --cpus-per-task in SLURM
N_JOBS = 32


# ===========================================================================
# Basic helpers
# ===========================================================================

def format_ci(lower, upper, digits=3):
    if np.isnan(lower) or np.isnan(upper):
        return ""
    return f"({lower:.{digits}f}, {upper:.{digits}f})"


def _safe_percentile(arr, q):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, q))


# ===========================================================================
# Bias-corrected and accelerated (BCa) bootstrap CI
# Reference: Efron & Tibshirani (1993), An Introduction to the Bootstrap,
#            Chapter 14.
# ===========================================================================

def _bca_ci(boot_values, observed, jackknife_values, alpha=0.05):
    """
    Compute BCa confidence interval.

    Parameters
    ----------
    boot_values      : array-like, shape (B,)
        Bootstrap replicate statistics.
    observed         : float
        Statistic computed on the original (full) sample.
    jackknife_values : array-like, shape (n,)
        Leave-one-out jackknife replicates of the statistic.
    alpha            : float
        Two-sided error level (default 0.05 -> 95% CI).

    Returns
    -------
    lower, upper : float
    """
    from scipy.stats import norm

    boot = np.asarray(boot_values, dtype=float)
    jack = np.asarray(jackknife_values, dtype=float)

    boot = boot[~np.isnan(boot)]
    jack = jack[~np.isnan(jack)]

    B = len(boot)
    if B == 0 or len(jack) == 0:
        return np.nan, np.nan

    # Bias-correction z0
    prop_less = np.mean(boot < observed)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    # Acceleration a (using jackknife)
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # Adjusted quantiles
    z_alpha_lo = norm.ppf(alpha / 2)
    z_alpha_hi = norm.ppf(1 - alpha / 2)

    def adj_quantile(z_alpha):
        num_ = z0 + z_alpha
        denom_ = 1 - a * (z0 + z_alpha)
        return norm.cdf(z0 + num_ / denom_)

    q_lo = adj_quantile(z_alpha_lo)
    q_hi = adj_quantile(z_alpha_hi)

    lower = _safe_percentile(boot, 100 * q_lo)
    upper = _safe_percentile(boot, 100 * q_hi)
    return lower, upper


# ===========================================================================
# Single bootstrap / jackknife iterations (run in parallel)
# ===========================================================================

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


def _one_jackknife_metric(i, y_true, y_pred, y_proba_arr, classes_):
    """Single leave-one-out jackknife iteration — runs in parallel."""
    mask = np.ones(len(y_true), dtype=bool)
    mask[i] = False
    yp = y_pred[mask]
    yt = y_true[mask]
    pp = None if y_proba_arr is None else y_proba_arr[mask, :]
    try:
        return calc_ordinal_metrics(yt, yp, pp, classes_=classes_)
    except Exception:
        return {m: np.nan for m in METRIC_NAMES}


# ===========================================================================
# Bootstrap metric CI (BCa) with raw bootstrap values saved
# ===========================================================================

def bootstrap_metric_ci(
    y_true, y_pred, y_proba=None, classes_=None,
    B=1000, random_state=123, n_jobs=N_JOBS,
):
    """
    BCa bootstrap CI for evaluation metrics.

    Confidence intervals use the bias-corrected and accelerated (BCa)
    method (Efron & Tibshirani, 1993, Chapter 14).

    Returns
    -------
    ci           : dict          — BCa CI bounds and formatted strings
    boot_df      : pd.DataFrame  — raw bootstrap values (B rows)
    jackknife_df : pd.DataFrame  — leave-one-out jackknife values (n rows)
    observed     : dict          — point estimates on the full sample
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba_arr = None if y_proba is None else np.asarray(y_proba)
    n = len(y_true)

    if classes_ is None:
        classes_ = np.sort(np.unique(y_true))

    # Point estimates on the full sample (needed for BCa bias-correction)
    observed = calc_ordinal_metrics(
        y_true, y_pred, y_proba_arr, classes_=classes_
    )

    # Bootstrap replicates
    boot_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_bootstrap_metric)(
            b, y_true, y_pred, y_proba_arr, n, classes_, random_state
        )
        for b in range(B)
    )
    boot_values = {m: [r[m] for r in boot_results] for m in METRIC_NAMES}
    boot_df = pd.DataFrame(boot_values)
    boot_df.index.name = "bootstrap_rep"

    # Jackknife replicates (for BCa acceleration parameter)
    jack_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_jackknife_metric)(
            i, y_true, y_pred, y_proba_arr, classes_
        )
        for i in range(n)
    )
    jack_values = {m: [r[m] for r in jack_results] for m in METRIC_NAMES}
    jackknife_df = pd.DataFrame(jack_values)
    jackknife_df.index.name = "jackknife_obs"

    # BCa CI for each metric
    ci = {}
    for m in METRIC_NAMES:
        lo, hi = _bca_ci(
            boot_values=boot_values[m],
            observed=observed[m],
            jackknife_values=jack_values[m],
        )
        ci[f"{m}_CI_Lower"] = lo
        ci[f"{m}_CI_Upper"] = hi
        ci[f"{m}_CI"] = format_ci(lo, hi)

    return ci, boot_df, jackknife_df, observed


def add_metric_ci_to_row(row, ci_dict):
    out = row.copy()
    for m in METRIC_NAMES:
        out[f"{m}_CI"] = ci_dict.get(f"{m}_CI", "")
        out[f"{m}_CI_Lower"] = ci_dict.get(f"{m}_CI_Lower", np.nan)
        out[f"{m}_CI_Upper"] = ci_dict.get(f"{m}_CI_Upper", np.nan)
    return out


# ===========================================================================
# Bootstrap permutation importance CI (BCa) with raw values saved
# ===========================================================================

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


def _one_jackknife_importance(
    i, fitted_model, X_np, y, predictors, metric_name, n_repeats, seed
):
    """Single leave-one-out jackknife for importance — runs in parallel."""
    mask = np.ones(len(y), dtype=bool)
    mask[i] = False
    X_df = pd.DataFrame(X_np[mask], columns=predictors)
    yj = y[mask]
    classes_ = np.array([0, 1, 2])
    rng = np.random.default_rng(seed + 99999 + i)

    base_pred = fitted_model.predict(X_df)
    try:
        base_proba = fitted_model.predict_proba(X_df)
    except Exception:
        base_proba = None
    base_metrics = calc_ordinal_metrics(yj, base_pred, base_proba, classes_=classes_)
    base_value = base_metrics.get(metric_name, np.nan)

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
            m_p = calc_ordinal_metrics(yj, pred_p, proba_p, classes_=classes_)
            perm_vals.append(m_p.get(metric_name, np.nan))
        importances[p] = base_value - float(np.nanmean(perm_vals))

    return importances


def permutation_importance_with_bootstrap_ci(
    fitted_model, X, y,
    metric_name="ORC",
    B=1000, n_repeats=10,
    random_state=123, n_jobs=N_JOBS,
):
    """
    BCa bootstrap CI for permutation variable importance.

    Confidence intervals use the bias-corrected and accelerated (BCa)
    method (Efron & Tibshirani, 1993, Chapter 14).

    Returns
    -------
    summary_df   : pd.DataFrame — one row per predictor with importance + BCa CI
    boot_imp_df  : pd.DataFrame — raw bootstrap importance values (B rows)
    jack_imp_df  : pd.DataFrame — jackknife importance values (n rows)
    """
    X = X.copy()
    y = np.asarray(y)
    predictors = list(X.columns)
    X_np = X.to_numpy()
    n = len(y)

    # Observed importance on the full sample
    rng0 = np.random.default_rng(random_state)
    classes_ = np.array([0, 1, 2])
    base_pred = fitted_model.predict(X)
    try:
        base_proba = fitted_model.predict_proba(X)
    except Exception:
        base_proba = None
    base_metrics = calc_ordinal_metrics(y, base_pred, base_proba, classes_=classes_)
    base_value = base_metrics.get(metric_name, np.nan)

    obs_importances = {}
    for p in predictors:
        pv = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[p] = rng0.permutation(Xp[p].to_numpy())
            pred_p = fitted_model.predict(Xp)
            try:
                proba_p = fitted_model.predict_proba(Xp)
            except Exception:
                proba_p = None
            m_p = calc_ordinal_metrics(y, pred_p, proba_p, classes_=classes_)
            pv.append(m_p.get(metric_name, np.nan))
        obs_importances[p] = base_value - float(np.nanmean(pv))

    # Bootstrap replicates
    boot_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_bootstrap_importance)(
            b, fitted_model, X_np, y, predictors,
            metric_name, n_repeats, random_state
        )
        for b in range(B)
    )
    all_imp = {p: [r[2][p] for r in boot_results] for p in predictors}
    boot_imp_df = pd.DataFrame(all_imp)
    boot_imp_df.index.name = "bootstrap_rep"

    # Jackknife replicates
    jack_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_jackknife_importance)(
            i, fitted_model, X_np, y, predictors,
            metric_name, n_repeats, random_state
        )
        for i in range(n)
    )
    jack_imp = {p: [r[p] for r in jack_results] for p in predictors}
    jack_imp_df = pd.DataFrame(jack_imp)
    jack_imp_df.index.name = "jackknife_obs"

    # BCa CI for each predictor
    rows = []
    for p in predictors:
        lo, hi = _bca_ci(
            boot_values=all_imp[p],
            observed=obs_importances[p],
            jackknife_values=jack_imp[p],
        )
        rows.append({
            "Predictor": p,
            f"Base_{metric_name}": base_value,
            f"Permuted_{metric_name}_Mean": float(np.nanmean(all_imp[p])),
            "Importance": obs_importances[p],
            "Importance_CI_Lower": lo,
            "Importance_CI_Upper": hi,
            "CI Importance": format_ci(lo, hi),
        })

    summary_df = pd.DataFrame(rows).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)

    return summary_df, boot_imp_df, jack_imp_df
