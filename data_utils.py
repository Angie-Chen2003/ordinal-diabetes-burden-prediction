from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AUTO_CREATE_BURDEN_SCORE_FROM_PAIN_VISION,
    BASE_DROP_COLS,
    COUNTRY_DROP_COLS,
    OUTCOME_CANDIDATES,
)


def normalize_country_key(country):
    """
    Standardize country names while preserving case.
    IMPORTANT: Do not use .upper(), because the project key is KLoSA, not KLOSA.
    """
    return str(country).strip()


def load_country_data(country, country_files):
    country = normalize_country_key(country)
    if country not in country_files:
        raise ValueError(
            f"Unknown country '{country}'. Available options: {list(country_files.keys())}"
        )

    path = Path(country_files[country])
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found for {country}: {path}\n"
            f"Please edit COUNTRY_FILES in config.py."
        )

    df = pd.read_excel(path)
    return df


def maybe_create_burden_score(df):
    df = df.copy()

    if "burden_score" in df.columns or "burden_score_int" in df.columns:
        return df

    if AUTO_CREATE_BURDEN_SCORE_FROM_PAIN_VISION:
        if "pain" in df.columns and "vision" in df.columns:
            df["pain"] = pd.to_numeric(df["pain"], errors="coerce")
            df["vision"] = pd.to_numeric(df["vision"], errors="coerce")
            df["burden_score"] = df["pain"] + df["vision"]

    return df


def find_outcome_column(df):
    for col in OUTCOME_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "No outcome column found. Expected one of: "
        f"{OUTCOME_CANDIDATES}. If needed, create burden_score before running."
    )


def prepare_xy(df, country):
    """
    Prepare X and y.

    Rules:
    - Auto-create burden_score from pain + vision if needed.
    - Keep ordinal categories 0, 1, 2.
    - Drop unavailable/non-predictor columns by country rules in config.py.
    - For POOL, country is label-encoded as a single integer column (not one-hot).
      This keeps 'country' as one predictor so permutation importance reports
      a single importance value for country membership.
    - All other categorical predictors are one-hot encoded.
    - Median-impute numeric predictors after encoding.
    """
    country = normalize_country_key(country)
    df = maybe_create_burden_score(df)
    outcome_col = find_outcome_column(df)

    df = df.copy()
    y_raw = df[outcome_col]

    if str(y_raw.dtype) == "category" or y_raw.dtype == object:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        if y_num.isna().all():
            y_num = pd.Series(y_raw).astype("category").cat.codes
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")

    df["_y_ordinal_"] = y_num
    df = df.dropna(subset=["_y_ordinal_"])

    df["_y_ordinal_"] = df["_y_ordinal_"].astype(int)
    df = df[df["_y_ordinal_"].isin([0, 1, 2])].copy()

    drop_cols = set(BASE_DROP_COLS)
    drop_cols.add(outcome_col)
    drop_cols.update(COUNTRY_DROP_COLS.get(country, []))
    drop_cols.add("_y_ordinal_")

    candidate_predictors = [c for c in df.columns if c not in drop_cols]

    # Label-encode the 'country' column BEFORE get_dummies so it stays
    # as a single integer column instead of being split into dummies.
    # This applies to Pool datasets where 'country' is kept as a predictor.
    if "country" in candidate_predictors:
        cat_series = df["country"].astype("category")
        country_label_map = dict(enumerate(cat_series.cat.categories))
        df["country"] = cat_series.cat.codes.astype(int)
        if country in ("Balanced_Pool", "Full_Pool"):
            print(f"  Label-encoded 'country' column: {country_label_map}")

    X = df[candidate_predictors].copy()
    y = df["_y_ordinal_"].astype(int).to_numpy()

    X = X.dropna(axis=1, how="all")

    # One-hot encode remaining categorical columns (excluding 'country'
    # which is now already numeric from label encoding above).
    cols_to_dummy = [
        c for c in X.columns
        if X[c].dtype == object or str(X[c].dtype) == "category"
    ]
    if cols_to_dummy:
        X = pd.get_dummies(X, columns=cols_to_dummy, drop_first=True, dummy_na=False)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.astype(float)

    nunique = X.nunique(dropna=True)
    keep_cols = [c for c in X.columns if nunique.get(c, 0) > 1]
    X = X[keep_cols]

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    X = X.fillna(0)

    return X, y


def align_columns(X_train, X_other):
    X_other = X_other.copy()
    for col in X_train.columns:
        if col not in X_other.columns:
            X_other[col] = 0
    X_other = X_other[X_train.columns]
    return X_other
