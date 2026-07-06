"""
run_pipeline.py
============================
Complete, fully-reproducible three-phase ordinal modelling pipeline.

Every country and pool dataset goes through full hyperparameter
grid search. Running this script on the same input datasets will
independently re-derive every best-parameter set via grid search,
making the full analysis reproducible end to end.

Confidence intervals use the bias-corrected and accelerated (BCa)
bootstrap method (Efron & Tibshirani, 1993, Chapter 14), with
B=1000 resamples. Raw bootstrap and jackknife replicates are saved
alongside the summary tables to allow downstream reanalysis.

Countries: SHARE, KLoSA, ELSA, HAALSI, CHARLS, HRS, LASI, MAHS
Pools:     Balanced_Pool, Full_Pool
Models:    Ordinal XGBoost, CORAL

Phase 1 — Pool training (grid search for both pools)
Phase 2 — Transfer evaluation (Pool models -> all 8 single countries)
Phase 3 — Single-country training (grid search for all 8 countries)
Phase 4 — Three-way comparison table

Output files (saved to OUTPUT_DIR):
    01_BalancedPool_train_test_with_CI.xlsx
    02_BalancedPool_table2_test_performance.xlsx
    03_BalancedPool_best_parameters.xlsx
    04_BalancedPool_cv_robustness.xlsx
    05_BalancedPool_variable_importance.xlsx
    06_BalancedPool_grid_search_results.xlsx
    07_BalancedPool_bootstrap_metric_raw.xlsx
    08_FullPool_train_test_with_CI.xlsx
    09_FullPool_table2_test_performance.xlsx
    10_FullPool_best_parameters.xlsx
    11_FullPool_cv_robustness.xlsx
    12_FullPool_variable_importance.xlsx
    13_FullPool_grid_search_results.xlsx
    14_FullPool_bootstrap_metric_raw.xlsx
    15_country_train_test_with_CI.xlsx
    16_country_table2_test_performance.xlsx
    17_country_best_parameters.xlsx
    18_country_cv_robustness.xlsx
    19_country_variable_importance.xlsx
    20_country_grid_search_results.xlsx
    21_country_bootstrap_metric_raw.xlsx
    22_transfer_all_pools_on_country_test_with_CI.xlsx
    23_comparison_threeway.xlsx
    24_transfer_bootstrap_metric_raw.xlsx

Runtime note:
    With 32 CPU cores, the full grid search for both pools plus all
    8 countries (x2 models each), followed by BCa bootstrap CI
    (1000 bootstrap + n jackknife iterations per evaluation) and
    permutation importance, is computationally heavy. Plan for
    several hours of wall time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BOOTSTRAP_B,
    CORAL_TUNING_GRID,
    COUNTRY_FILES,
    CV_N_SPLITS,
    OUTPUT_DIR,
    PERMUTATION_N_REPEATS,
    POOL_FILES,
    PRIMARY_IMPORTANCE_METRIC,
    PRIMARY_SELECTION_METRIC,
    RANDOM_STATE,
    TEST_SIZE,
    XGB_TUNING_GRID,
)
from cv_robustness import cross_validation_robustness
from data_utils import load_country_data, normalize_country_key, prepare_xy
from metrics import METRIC_NAMES, calc_ordinal_metrics, metrics_to_row
from tuning import fit_final_and_evaluate, grid_search_tune_model
from uncertainty import (
    add_metric_ci_to_row,
    bootstrap_metric_ci,
    permutation_importance_with_bootstrap_ci,
)

# ===========================================================================
# Configuration
# ===========================================================================

ALL_COUNTRIES = ["SHARE", "KLoSA", "ELSA", "HAALSI", "CHARLS", "HRS", "LASI", "MAHS"]

MODEL_SPECS = {
    "Ordinal XGBoost": XGB_TUNING_GRID,
    "CORAL": CORAL_TUNING_GRID,
}

# Every (country/pool, model) combination is tuned via
# grid_search_tune_model() below.


# ===========================================================================
# Helpers
# ===========================================================================

def _empty_outputs():
    return {
        "train_validation_rows": [],
        "table2_rows": [],
        "parameter_rows": [],
        "cv_summary_rows": [],
        "cv_fold_rows": [],
        "vi_tables": {},
        "grid_search_tables": {},
        "boot_metric_tables": {},
    }


def _make_train_row(country, model_name, train_metrics):
    row = {"Country": country,
           **metrics_to_row(model_name, "Train", train_metrics)}
    for m in METRIC_NAMES:
        row[f"{m}_CI"] = ""
        row[f"{m}_CI_Lower"] = np.nan
        row[f"{m}_CI_Upper"] = np.nan
    return row


def _make_test_row_and_ci(country, model_name, dataset_label,
                           test_metrics, y_test, pred_test, proba_test):
    base = {"Country": country,
            **metrics_to_row(model_name, dataset_label, test_metrics)}
    ci, boot_df, jack_df, _ = bootstrap_metric_ci(
        y_true=y_test, y_pred=pred_test, y_proba=proba_test,
        classes_=np.array([0, 1, 2]),
        B=BOOTSTRAP_B, random_state=RANDOM_STATE,
    )
    return add_metric_ci_to_row(base, ci), ci, boot_df


def _make_table2_row(country, model_name, eval_type, test_metrics, ci):
    row = {"Country": country, "Model": model_name, "Evaluation_Type": eval_type}
    for m in METRIC_NAMES:
        row[m] = test_metrics[m]
        row[f"{m}_CI"] = ci[f"{m}_CI"]
        row[f"{m}_CI_Lower"] = ci[f"{m}_CI_Lower"]
        row[f"{m}_CI_Upper"] = ci[f"{m}_CI_Upper"]
    return row


def _run_full_evaluation(
    country, model_name, best_params,
    X_train, y_train, X_test, y_test,
    X_full, y_full, outputs,
    used_grid_search=True, grid_df=None,
):
    """Train, evaluate, CV robustness, variable importance for one country+model."""
    final = fit_final_and_evaluate(
        model_name=model_name, best_params=best_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )

    # Train row (no CI)
    outputs["train_validation_rows"].append(
        _make_train_row(country, model_name, final["train_metrics"])
    )

    # Test row + BCa CI + raw bootstrap values
    test_row, ci, boot_df = _make_test_row_and_ci(
        country, model_name, "Test",
        final["test_metrics"], y_test,
        final["test_pred"], final["test_proba"],
    )
    outputs["train_validation_rows"].append(test_row)
    outputs["table2_rows"].append(
        _make_table2_row(country, model_name, "Country_Specific",
                         final["test_metrics"], ci)
    )

    # Save raw bootstrap metric values
    boot_df.insert(0, "Country", country)
    boot_df.insert(1, "Model", model_name)
    boot_df.insert(2, "Split", "Test")
    outputs["boot_metric_tables"][f"{country}_{model_name}"] = boot_df

    # Parameters
    param_row = {
        "Country": country,
        "Model": model_name,
        "Used_Grid_Search": used_grid_search,
    }
    param_row.update(best_params)
    outputs["parameter_rows"].append(param_row)

    # Grid search results
    if grid_df is not None:
        outputs["grid_search_tables"][f"{country}_{model_name}"] = grid_df

    # CV robustness
    cv_summary, cv_folds = cross_validation_robustness(
        model_name=model_name, params=best_params,
        X=X_full, y=y_full,
        n_splits=CV_N_SPLITS, random_state=RANDOM_STATE,
    )
    cv_summary.insert(0, "Country", country)
    cv_folds.insert(0, "Country", country)
    outputs["cv_summary_rows"].extend(cv_summary.to_dict("records"))
    outputs["cv_fold_rows"].append(cv_folds)

    # Variable importance + BCa CI
    vi_df, vi_boot_df, vi_jack_df = permutation_importance_with_bootstrap_ci(
        fitted_model=final["model"],
        X=X_test, y=y_test,
        metric_name=PRIMARY_IMPORTANCE_METRIC,
        B=BOOTSTRAP_B, n_repeats=PERMUTATION_N_REPEATS,
        random_state=RANDOM_STATE,
    )
    vi_df.insert(0, "Country", country)
    vi_df.insert(1, "Model", model_name)
    outputs["vi_tables"][f"{country}_{model_name}"] = vi_df

    return final["model"]


def _save_xlsx(path, sheet_dict, group_by=None):
    if not sheet_dict:
        sheet_dict = {"Empty": pd.DataFrame({"Note": ["No data."]})}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheet_dict.items():
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                pd.DataFrame().to_excel(
                    writer, sheet_name=str(sheet_name)[:31], index=False)
                continue
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
            if group_by and sheet_name == "All" and group_by in df.columns:
                for grp_val, sub in df.groupby(group_by):
                    safe = str(grp_val)[:31]
                    if safe not in [str(k)[:31] for k in sheet_dict.keys()]:
                        sub.to_excel(writer, sheet_name=safe, index=False)
    print(f"  Saved: {path.name}")


# ===========================================================================
# Phase 1: Pool training — grid search for BOTH pools
# ===========================================================================

def run_pool_phase():
    print("\n" + "=" * 80)
    print("PHASE 1: POOL TRAINING (full grid search, both pools)")
    print("=" * 80)

    fitted_pool_models = {}
    pool_feature_cols = {}
    file_offsets = {name: 1 + i * 7
                    for i, name in enumerate(POOL_FILES.keys())}

    for pool_name, pool_file in POOL_FILES.items():
        print(f"\n{'='*60}\n  Pool: {pool_name}\n{'='*60}")
        outputs = _empty_outputs()
        fitted_pool_models[pool_name] = {}

        pool_df = pd.read_excel(pool_file)
        X_pool, y_pool = prepare_xy(pool_df, pool_name)
        print(f"  X shape = {X_pool.shape}")
        print(f"  y = {pd.Series(y_pool).value_counts().sort_index().to_dict()}")

        if X_pool.shape[1] == 0:
            raise ValueError(f"No predictors for {pool_name}.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_pool, y_pool,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_pool,
        )
        pool_feature_cols[pool_name] = X_train.columns.tolist()

        for model_name, grid in MODEL_SPECS.items():
            print(f"\n  {pool_name} | {model_name} (grid search)")
            try:
                best_params, grid_df = grid_search_tune_model(
                    model_name=model_name,
                    X_train_full=X_train, y_train_full=y_train,
                    param_grid=grid,
                    random_state=RANDOM_STATE,
                    primary_metric=PRIMARY_SELECTION_METRIC,
                )
                print(f"  Best params: {best_params}")

                fitted_model = _run_full_evaluation(
                    country=pool_name, model_name=model_name,
                    best_params=best_params,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    X_full=X_pool, y_full=y_pool,
                    outputs=outputs,
                    used_grid_search=True, grid_df=grid_df,
                )
                fitted_pool_models[pool_name][model_name] = fitted_model

            except Exception as e:
                print(f"  Skipping {model_name} for {pool_name}: {repr(e)}")

        offset = file_offsets[pool_name]
        tag = pool_name.replace("_", "")
        _save_xlsx(OUTPUT_DIR / f"{offset:02d}_{tag}_train_test_with_CI.xlsx",
                   {"All": pd.DataFrame(outputs["train_validation_rows"])})
        _save_xlsx(OUTPUT_DIR / f"{offset+1:02d}_{tag}_table2_test_performance.xlsx",
                   {"All": pd.DataFrame(outputs["table2_rows"])})
        _save_xlsx(OUTPUT_DIR / f"{offset+2:02d}_{tag}_best_parameters.xlsx",
                   {"All": pd.DataFrame(outputs["parameter_rows"])})
        cv_s = pd.DataFrame(outputs["cv_summary_rows"])
        cv_f = (pd.concat(outputs["cv_fold_rows"], ignore_index=True)
                if outputs["cv_fold_rows"] else pd.DataFrame())
        _save_xlsx(OUTPUT_DIR / f"{offset+3:02d}_{tag}_cv_robustness.xlsx",
                   {"Summary": cv_s, "Fold-level": cv_f})
        _save_xlsx(OUTPUT_DIR / f"{offset+4:02d}_{tag}_variable_importance.xlsx",
                   outputs["vi_tables"])
        _save_xlsx(OUTPUT_DIR / f"{offset+5:02d}_{tag}_grid_search_results.xlsx",
                   outputs["grid_search_tables"])
        _save_xlsx(OUTPUT_DIR / f"{offset+6:02d}_{tag}_bootstrap_metric_raw.xlsx",
                   outputs["boot_metric_tables"])

    return fitted_pool_models, pool_feature_cols


# ===========================================================================
# Phase 2: Transfer evaluation
# ===========================================================================

def run_transfer_phase(fitted_pool_models, pool_feature_cols):
    print("\n" + "=" * 80)
    print("PHASE 2: TRANSFER EVALUATION (Pool models -> all 8 countries)")
    print("=" * 80)

    transfer_rows = []
    table2_rows = []
    transfer_boot_tables = {}

    for pool_name, models in fitted_pool_models.items():
        for country in ALL_COUNTRIES:
            country = normalize_country_key(country)
            print(f"\n  {pool_name} -> {country}")

            try:
                df_raw = load_country_data(country, COUNTRY_FILES)
                X_country, y_country = prepare_xy(df_raw, country)
            except Exception as e:
                print(f"  Skipping {country}: {repr(e)}")
                continue

            _, X_test, _, y_test = train_test_split(
                X_country, y_country,
                test_size=TEST_SIZE, random_state=RANDOM_STATE,
                stratify=y_country,
            )

            feat_cols = pool_feature_cols[pool_name]
            X_aligned = X_test.copy()
            for col in feat_cols:
                if col not in X_aligned.columns:
                    X_aligned[col] = 0
            X_aligned = X_aligned[feat_cols]
            for col in X_aligned.columns:
                X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce")
            X_aligned = X_aligned.fillna(0)

            for model_name, pool_model in models.items():
                try:
                    pred = pool_model.predict(X_aligned)
                    try:
                        proba = pool_model.predict_proba(X_aligned)
                    except Exception:
                        proba = None

                    test_metrics = calc_ordinal_metrics(
                        y_test, pred, proba, classes_=np.array([0, 1, 2]))

                    eval_label = f"{pool_name}_Transfer"
                    base_row = {
                        "Country": country,
                        "Pool": pool_name,
                        "Evaluation_Type": eval_label,
                        **metrics_to_row(model_name, "Transfer_Test", test_metrics),
                    }

                    ci, boot_df, jack_df, _ = bootstrap_metric_ci(
                        y_true=y_test, y_pred=pred, y_proba=proba,
                        classes_=np.array([0, 1, 2]),
                        B=BOOTSTRAP_B, random_state=RANDOM_STATE,
                    )
                    transfer_rows.append(add_metric_ci_to_row(base_row, ci))

                    t2 = _make_table2_row(
                        country, model_name, eval_label, test_metrics, ci)
                    t2["Pool"] = pool_name
                    table2_rows.append(t2)

                    # Save raw bootstrap values
                    boot_df.insert(0, "Country", country)
                    boot_df.insert(1, "Pool", pool_name)
                    boot_df.insert(2, "Model", model_name)
                    transfer_boot_tables[f"{pool_name}_{country}_{model_name}"] = boot_df

                except Exception as e:
                    print(f"  FAILED {model_name}: {repr(e)}")

    return {
        "transfer_rows": transfer_rows,
        "table2_rows": table2_rows,
        "transfer_boot_tables": transfer_boot_tables,
    }


# ===========================================================================
# Phase 3: Single-country training — grid search for ALL 8 countries
# ===========================================================================

def run_country_phase():
    print("\n" + "=" * 80)
    print("PHASE 3: SINGLE-COUNTRY TRAINING (full grid search, all 8 countries)")
    print("=" * 80)

    outputs = _empty_outputs()

    for country in ALL_COUNTRIES:
        country = normalize_country_key(country)
        print(f"\n{'='*60}")
        print(f"  Country: {country} (grid search)")
        print("=" * 60)

        try:
            df_raw = load_country_data(country, COUNTRY_FILES)
            X, y = prepare_xy(df_raw, country)
        except Exception as e:
            print(f"  Skipping {country}: {repr(e)}")
            continue

        print(f"  X shape = {X.shape}")
        print(f"  y = {pd.Series(y).value_counts().sort_index().to_dict()}")

        if X.shape[1] == 0:
            print(f"  WARNING: No predictors for {country}. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y,
        )

        for model_name, grid in MODEL_SPECS.items():
            print(f"\n  {country} | {model_name} (grid search)")
            try:
                best_params, grid_df = grid_search_tune_model(
                    model_name=model_name,
                    X_train_full=X_train, y_train_full=y_train,
                    param_grid=grid,
                    random_state=RANDOM_STATE,
                    primary_metric=PRIMARY_SELECTION_METRIC,
                )
                print(f"  Best params: {best_params}")
                _run_full_evaluation(
                    country=country, model_name=model_name,
                    best_params=best_params,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    X_full=X, y_full=y,
                    outputs=outputs,
                    used_grid_search=True, grid_df=grid_df,
                )
            except Exception as e:
                print(f"  Skipping {model_name}: {repr(e)}")

    return outputs


def save_country_outputs(outputs):
    _save_xlsx(OUTPUT_DIR / "15_country_train_test_with_CI.xlsx",
               {"All": pd.DataFrame(outputs["train_validation_rows"])},
               group_by="Country")
    _save_xlsx(OUTPUT_DIR / "16_country_table2_test_performance.xlsx",
               {"All": pd.DataFrame(outputs["table2_rows"])},
               group_by="Country")
    _save_xlsx(OUTPUT_DIR / "17_country_best_parameters.xlsx",
               {"All": pd.DataFrame(outputs["parameter_rows"])},
               group_by="Model")
    cv_s = pd.DataFrame(outputs["cv_summary_rows"])
    cv_f = (pd.concat(outputs["cv_fold_rows"], ignore_index=True)
            if outputs["cv_fold_rows"] else pd.DataFrame())
    _save_xlsx(OUTPUT_DIR / "18_country_cv_robustness.xlsx",
               {"Summary": cv_s, "Fold-level": cv_f})
    _save_xlsx(OUTPUT_DIR / "19_country_variable_importance.xlsx",
               outputs["vi_tables"])
    _save_xlsx(OUTPUT_DIR / "20_country_grid_search_results.xlsx",
               outputs["grid_search_tables"])
    _save_xlsx(OUTPUT_DIR / "21_country_bootstrap_metric_raw.xlsx",
               outputs["boot_metric_tables"])


# ===========================================================================
# Phase 4: Three-way comparison
# ===========================================================================

def build_threeway_comparison(transfer_table2, country_table2_rows):
    transfer_df = pd.DataFrame(transfer_table2)
    country_df = pd.DataFrame(country_table2_rows)

    if transfer_df.empty or country_df.empty:
        print("WARNING: Cannot build comparison — missing data.")
        return pd.DataFrame()

    pool_names = (transfer_df["Pool"].unique().tolist()
                  if "Pool" in transfer_df.columns else [])
    rows = []

    for model_name in country_df["Model"].unique():
        for country in country_df["Country"].unique():
            c_sub = country_df[
                (country_df["Model"] == model_name) &
                (country_df["Country"] == country)
            ]
            if c_sub.empty:
                continue
            c_row = c_sub.iloc[0]

            pool_rows = {}
            for pool_name in pool_names:
                t_sub = transfer_df[
                    (transfer_df["Model"] == model_name) &
                    (transfer_df["Country"] == country) &
                    (transfer_df["Pool"] == pool_name)
                ]
                if not t_sub.empty:
                    pool_rows[pool_name] = t_sub.iloc[0]

            for m in METRIC_NAMES:
                out = {
                    "Country": country,
                    "Model": model_name,
                    "Metric": m,
                    "Country_Specific_Value": c_row.get(m, np.nan),
                    "Country_Specific_CI": c_row.get(f"{m}_CI", ""),
                }
                for pool_name, p_row in pool_rows.items():
                    tag = pool_name.replace("_", "")
                    p_val = p_row.get(m, np.nan)
                    out[f"{tag}_Transfer_Value"] = p_val
                    out[f"{tag}_Transfer_CI"] = p_row.get(f"{m}_CI", "")
                    try:
                        out[f"Diff_{tag}_minus_Country"] = (
                            float(p_val) - float(c_row.get(m, np.nan))
                        )
                    except Exception:
                        out[f"Diff_{tag}_minus_Country"] = np.nan

                if len(pool_rows) == 2:
                    pnames = list(pool_rows.keys())
                    try:
                        v0 = float(pool_rows[pnames[0]].get(m, np.nan))
                        v1 = float(pool_rows[pnames[1]].get(m, np.nan))
                        t0 = pnames[0].replace("_", "")
                        t1 = pnames[1].replace("_", "")
                        out[f"Diff_{t0}_minus_{t1}"] = v0 - v1
                    except Exception:
                        pass

                rows.append(out)

    return pd.DataFrame(rows)


# ===========================================================================
# Main
# ===========================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Phase 1: Pool training (grid search)
    fitted_pool_models, pool_feature_cols = run_pool_phase()

    # Phase 2: Transfer evaluation
    transfer_result = run_transfer_phase(fitted_pool_models, pool_feature_cols)
    _save_xlsx(
        OUTPUT_DIR / "22_transfer_all_pools_on_country_test_with_CI.xlsx",
        {"All": pd.DataFrame(transfer_result["transfer_rows"])},
        group_by="Country",
    )
    _save_xlsx(
        OUTPUT_DIR / "24_transfer_bootstrap_metric_raw.xlsx",
        transfer_result["transfer_boot_tables"],
    )

    # Phase 3: Single-country training (grid search)
    country_outputs = run_country_phase()
    save_country_outputs(country_outputs)

    # Phase 4: Three-way comparison
    comparison_df = build_threeway_comparison(
        transfer_table2=transfer_result["table2_rows"],
        country_table2_rows=country_outputs["table2_rows"],
    )
    _save_xlsx(
        OUTPUT_DIR / "23_comparison_threeway.xlsx",
        {"Comparison": comparison_df},
    )

    print("\n" + "=" * 80)
    print("ALL DONE. Output files:")
    for p in sorted(OUTPUT_DIR.glob("*.xlsx")):
        print(f"  {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
