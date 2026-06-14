import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, train_test_split

from config import INNER_VALIDATION_SIZE, PRIMARY_SELECTION_METRIC, RANDOM_STATE
from metrics import METRIC_NAMES, calc_ordinal_metrics
from models import make_model

N_JOBS = 32


def clean_xgb_params(params):
    params = params.copy()
    params.pop("config_note", None)
    if "n_estimator" in params and "n_estimators" not in params:
        params["n_estimators"] = params.pop("n_estimator")
    return params


def clean_coral_params(params):
    params = params.copy()
    params.pop("config_note", None)
    return params


def clean_params(model_name, params):
    model_name = str(model_name).upper()
    if "XGB" in model_name:
        return clean_xgb_params(params)
    if "CORAL" in model_name:
        return clean_coral_params(params)
    return params.copy()


def compare_sort_key(row, primary_metric=PRIMARY_SELECTION_METRIC):
    return (
        row.get(primary_metric, -np.inf),
        row.get("Macro_F1", -np.inf),
        row.get("Accuracy", -np.inf),
        -row.get("MAE", np.inf),
    )


def _evaluate_one_config(i, raw_params, model_name, X_subtrain, y_subtrain, X_valid, y_valid):
    """Evaluate one grid search config — runs in parallel."""
    params = clean_params(model_name, raw_params)
    start = time.time()
    status = "OK"
    err_msg = ""

    try:
        model = make_model(model_name, params)
        model.fit(X_subtrain, y_subtrain)
        pred = model.predict(X_valid)
        try:
            proba = model.predict_proba(X_valid)
        except Exception:
            proba = None
        metrics = calc_ordinal_metrics(
            y_valid, pred, y_proba=proba, classes_=np.array([0, 1, 2])
        )
    except Exception as e:
        status = "FAILED"
        err_msg = repr(e)
        metrics = {m: np.nan for m in METRIC_NAMES}

    elapsed = time.time() - start
    row = {
        "Model": model_name,
        "Grid_Index": i,
        "Status": status,
        "Error": err_msg,
        "Elapsed_seconds": elapsed,
    }
    row.update(raw_params)
    row.update(metrics)
    return row, params, metrics, status


def grid_search_tune_model(
    model_name,
    X_train_full,
    y_train_full,
    param_grid,
    random_state=RANDOM_STATE,
    primary_metric=PRIMARY_SELECTION_METRIC,
    n_jobs=N_JOBS,
):
    """
    Parallel full grid search.
    All parameter configurations are evaluated simultaneously across CPU cores.
    """
    X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
        X_train_full, y_train_full,
        test_size=INNER_VALIDATION_SIZE,
        random_state=random_state,
        stratify=y_train_full,
    )

    grid = list(ParameterGrid(param_grid))
    print(f"\n[{model_name}] Grid search configs: {len(grid)} (parallel, n_jobs={n_jobs})")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_evaluate_one_config)(
            i + 1, raw_params, model_name,
            X_subtrain, y_subtrain, X_valid, y_valid
        )
        for i, raw_params in enumerate(grid)
    )

    records = []
    best_score_key = None
    best_params = None

    for row, params, metrics, status in results:
        records.append(row)
        if status == "OK":
            this_key = compare_sort_key(metrics, primary_metric=primary_metric)
            if best_score_key is None or this_key > best_score_key:
                best_score_key = this_key
                best_params = params

    print(f"  finished {len(grid)}/{len(grid)}")
    results_df = pd.DataFrame(records)

    if best_params is None:
        raise RuntimeError(f"All grid-search configs failed for {model_name}.")

    return best_params, results_df


def fit_final_and_evaluate(model_name, best_params, X_train, y_train, X_test, y_test):
    model = make_model(model_name, best_params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    try:
        proba_train = model.predict_proba(X_train)
    except Exception:
        proba_train = None

    try:
        proba_test = model.predict_proba(X_test)
    except Exception:
        proba_test = None

    train_metrics = calc_ordinal_metrics(
        y_train, pred_train, y_proba=proba_train, classes_=np.array([0, 1, 2])
    )
    test_metrics = calc_ordinal_metrics(
        y_test, pred_test, y_proba=proba_test, classes_=np.array([0, 1, 2])
    )

    return {
        "model": model,
        "train_pred": pred_train,
        "train_proba": proba_train,
        "test_pred": pred_test,
        "test_proba": proba_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
