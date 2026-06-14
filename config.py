from pathlib import Path
# ======================================================================
# This part is for loading datasets, preparation and tuning grid setting 
# ======================================================================

# Folder containing the two pooled datasets (Balanced_Pool, Full_Pool)
POOL_DATA_DIR = Path("/path/to/data/Pool pipeline")

# Folder containing the 8 single-country datasets
COUNTRY_DATA_DIR = Path("/path/to/data/Pool pipeline/Pool datasets")

# Folder where all result tables will be written
OUTPUT_DIR = Path("/path/to/pipeline/outputs")


OUTPUT_DIR.mkdir(exist_ok=True)
PROJECT_DIR = Path(__file__).resolve().parent

POOL_FILES = {
    "Balanced_Pool": POOL_DATA_DIR / "Balanced_Pooled_dataset.xlsx",
    "Full_Pool":     POOL_DATA_DIR / "Full_Pooled_dataset.xlsx",
}

COUNTRY_FILES = {
    "SHARE": COUNTRY_DATA_DIR / "SHARE_final.xlsx",
    "KLoSA": COUNTRY_DATA_DIR / "KLoSA_final.xlsx",
    "ELSA": COUNTRY_DATA_DIR / "ELSA_final.xlsx",
    "HAALSI": COUNTRY_DATA_DIR / "HAALSI_final.xlsx",
    "CHARLS": COUNTRY_DATA_DIR / "CHARLS_final.xlsx",
    "HRS": COUNTRY_DATA_DIR / "HRS_final.xlsx",
    "LASI": COUNTRY_DATA_DIR / "LASI_final.xlsx",
    "MAHS": COUNTRY_DATA_DIR / "MAHS_final.xlsx",
}

# ============================================================
# 1. Variable rules
# ============================================================

OUTCOME_CANDIDATES = [
    "burden_score_int",
    "burden_score",
    "burden_score_ord",
    "outcome",
    "Y",
    "y",
]

BASE_DROP_COLS = [
    "burden_score",
    "burden_score_int",
    "burden_score_ord",
    "pain",
    "vision",
    "mergeid",
    "id",
    "ID",
]

# "country" and "social_eco_ppp" are for generating pooled dataset.
# "medical_insurance" is the necessary variable for separate dataset.

# Since some separate countries have fixed constant columns, like KLoSA
# and ELSA, which are all have a "medical_insurance" with value "1". So
# for ensuring the dataset is full ranked when training themselves, we
# used "COUNTRY_DROP_COLS" to erase "medical_insurance".

# Also, since our final dataset for each separate country is well-preparaed
# for merging full pooled dataset, some variables like "country" and "social_eco_ppp"
# occur. For avioding the "no_full_rank", these two variables are also deleted for
# training separate country successfully.
COUNTRY_DROP_COLS = {
    "KLoSA": ["medical_insurance", "country", "social_eco_ppp"],
    "SHARE": ["country", "social_eco_ppp"],
    "ELSA": ["medical_insurance", "country", "social_eco_ppp"],
    "HAALSI": ["country", "social_eco_ppp"],
    "CHARLS": ["country", "social_eco_ppp"],
    "HRS": ["country", "social_eco_ppp"],
    "LASI": ["country", "social_eco_ppp"],
    "MAHS": ["country", "social_eco_ppp"],
    "Balanced_Pool": ["social_eco"],
    "Full_Pool": ["social_eco"],
}

AUTO_CREATE_BURDEN_SCORE_FROM_PAIN_VISION = True

# ============================================================
# 2. Reproducibility and split settings
# ============================================================

RANDOM_STATE = 123
TEST_SIZE = 0.20
INNER_VALIDATION_SIZE = 0.20
CV_N_SPLITS = 5

BOOTSTRAP_B = 1000
PERMUTATION_N_REPEATS = 10

PRIMARY_SELECTION_METRIC = "ORC"
PRIMARY_IMPORTANCE_METRIC = "ORC"

# =================================================================================
# 3. Model tuning grids, you can add your own new candidate parameters in this grid
# =================================================================================

XGB_TUNING_GRID = [
    {
        "aggregation": ["weighted"],
        "norm": [True],
        "learning_rate": [0.01, 0.03, 0.05, 0.10],
        "max_depth": [2, 3, 4],
        "n_estimators": [100, 150, 300],
        "subsample": [0.70, 0.85, 1.00],
        "colsample_bytree": [0.70, 0.85, 1.00],
        "reg_lambda": [1, 5, 10],
        "gamma": [0, 1],
        "min_child_weight": [1, 5],
        "random_state": [RANDOM_STATE],
        "eval_metric": ["logloss"],
    }
]

CORAL_TUNING_GRID = [
    {
        "hidden1": [16, 32, 64],
        "hidden2": [0, 16, 32],
        "dropout": [0.00, 0.20, 0.40],
        "lr": [0.001, 0.003, 0.01],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "batch_size": [32, 64],
        "epochs": [80, 150],
        "random_state": [RANDOM_STATE],
    }
]
