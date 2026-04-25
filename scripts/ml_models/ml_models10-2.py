# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15
#
# CHANGES FROM ml_models9.py:
#   1. NEW INTERACTION FEATURES
#        carrier_month_enc → target-encoded carrier×month combination.
#          Carrier delay rates swing 8-12% between summer and winter —
#          a single carrier_delay_rate misses this seasonal variation.
#        route_dep_period_enc → target-encoded route×dep_period.
#          A late-night ORD→LAX is systematically different from the 7am
#          departure: crew positioning, overnight maintenance, cascading
#          delays from earlier in the day all vary by departure window.
#        is_early_morning → binary flag for pre-6am flights. These are
#          often the first departure of the day so they can't inherit
#          upstream delay, but maintenance issues surface here.
#
#   2. 3-WAY ENSEMBLE (XGB + LGBM + CatBoost) — the v9 equal/weighted
#      blend of XGB+CB UNDER-performed CatBoost alone (0.7091 vs 0.7096)
#      because the two models are too correlated (target-encoded floats
#      often produce similar splits). LightGBM uses leaf-wise growth vs
#      CatBoost's oblivious trees — adding it introduces genuine diversity.
#
#   3. STACKING META-LEARNER — instead of naive averaging, train a
#      Logistic Regression on out-of-fold predictions of the 3 base models.
#      The meta-learner learns optimal weighting from data rather than
#      guessing (equal or AUC-weighted). Uses 5-fold CV on training set.
#
#   4. OPTUNA HYPERPARAMETER TUNING — CatBoost is the performance leader
#      so we tune it with 30 Optuna trials (depth, learning_rate,
#      l2_leaf_reg, colsample_bylevel). The best params are then used for
#      the full 6.4M run.
#      Install: pip install optuna
#
#   dep_del15 intentionally excluded — pre-departure prediction framing
# =============================================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
    precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not found. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not found. Run: pip install lightgbm")
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not found. Run: pip install catboost")
    CATBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not found. Run: pip install optuna  (skipping hyperparameter tuning)")
    OPTUNA_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
# Set working directory to repo root so all relative paths resolve correctly
import pathlib as _pl, os as _os
_os.chdir(_pl.Path(__file__).resolve().parent.parent.parent)

# Ensure output directories exist (safe on fresh clones)
_os.makedirs("charts", exist_ok=True)
_os.makedirs("outputs/catboost_info", exist_ok=True)

DB_PATH = "data/flights.db"

conn = sqlite3.connect(DB_PATH)
query = """
    SELECT
        f.flight_date,
        f.origin,
        f.dest,
        f.distance,
        f.crs_dep_time,
        f.carrier,
        f.dep_del15,
        f.arr_del15,
        a.state AS origin_state
    FROM flight f
    LEFT JOIN airport a ON f.origin = a.iata_code
    WHERE f.arr_del15 IS NOT NULL
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df):,} rows")
float_cols = df.select_dtypes(include="float64").columns
df[float_cols] = df[float_cols].astype("float32")
print(f"Memory usage after downcast: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Don't copy the full 2GB DataFrame into df_full — it sits in RAM unused
# for the entire exploration run and causes XGBoost to OOM.
# Re-query the DB in Step 7 instead (fast — SQLite read is the bottleneck,
# not Python, and it avoids holding two large DataFrames simultaneously).
df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows (exploration run)")
print(f"\nClass balance:\n{df['arr_del15'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

_US_HOLIDAY_MD = [
    (1, 1), (1, 15), (2, 19), (5, 27), (7, 4),
    (9, 2), (11, 11), (11, 28), (12, 25), (12, 31),
]

def _near_holiday(date, window=2):
    for m, d in _US_HOLIDAY_MD:
        try:
            h = pd.Timestamp(date.year, m, d)
        except ValueError:
            continue
        if abs((date - h).days) <= window:
            return 1
    return 0


def engineer_features(df):
    df = df.copy()

    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df["month"]       = df["flight_date"].dt.month
    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["dep_hour"] = pd.to_datetime(
        df["crs_dep_time"], format="%H:%M:%S", errors="coerce"
    ).dt.hour

    df["dep_period"] = pd.cut(
        df["dep_hour"], bins=[-1, 5, 11, 16, 20, 23], labels=[0, 1, 2, 3, 4]
    ).astype("Int64").fillna(2).astype(int)

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["dep_hour"]    / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["dep_hour"]    / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Log-distance (raw distance dropped — collinear)
    df["log_distance"] = np.log1p(df["distance"].astype(float))

    # Calendar context
    df["near_holiday"]    = df["flight_date"].apply(_near_holiday)
    df["is_peak"]         = df["month"].isin([6, 7, 8, 12]).astype(int)
    df["is_early_morning"] = (df["dep_hour"] < 6).astype(int)  # NEW

    # Base interaction keys
    df["route"]          = df["origin"].astype(str) + "_" + df["dest"].astype(str)
    df["carrier_origin"] = df["carrier"].astype(str) + "_" + df["origin"].astype(str)
    df["origin_state"]   = df["origin_state"].fillna("UNKNOWN")

    # NEW interaction features for target encoding
    # carrier × month — captures seasonal variation in carrier performance
    df["carrier_month"]      = df["carrier"].astype(str) + "_" + df["month"].astype(str)
    # route × dep_period — early-morning vs evening departures differ systematically
    df["route_dep_period"]   = df["route"].astype(str) + "_" + df["dep_period"].astype(str)

    return df


df = engineer_features(df)


# =============================================================================
# STEP 3: ENCODE CATEGORICALS (XGB / LGBM / LR / RF)
# =============================================================================

def encode_categoricals(df):
    encoders = {}
    for col, key in [
        ("dest",             "dest_enc"),
        ("origin_state",     "state_enc"),
        ("carrier",          "carrier_enc"),
        ("route",            "route_enc"),
        ("carrier_origin",   "carrier_origin_enc"),
        ("carrier_month",    "carrier_month_enc"),     # NEW
        ("route_dep_period", "route_dep_period_enc"),  # NEW
    ]:
        le = LabelEncoder()
        df[key] = le.fit_transform(df[col].fillna("UNKNOWN").astype(str))
        encoders[key] = le
    return df, encoders

df, encoders = encode_categoricals(df)


# =============================================================================
# STEP 4: TRAIN / TEST SPLIT
# =============================================================================
TARGET = "arr_del15"

FEATURES_BASE = [
    "month", "day_of_week", "is_weekend", "dep_hour", "dep_period",
    "log_distance",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "near_holiday", "is_peak", "is_early_morning",
    "dest_enc", "carrier_enc", "route_enc", "carrier_origin_enc",
    "carrier_month_enc", "route_dep_period_enc",
]

CAT_COLS_CB = ["dest", "carrier", "route", "carrier_origin", "origin_state",
               "carrier_month", "route_dep_period"]
FEATURES_CB = [
    "month", "day_of_week", "is_weekend", "dep_hour", "dep_period",
    "log_distance",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "near_holiday", "is_peak", "is_early_morning",
] + CAT_COLS_CB

all_cols = list(dict.fromkeys(FEATURES_BASE + FEATURES_CB + [TARGET]))
model_df = df[all_cols].dropna()

X_all = model_df.drop(columns=[TARGET])
y     = model_df[TARGET].astype(int)

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

X_train    = X_train_all[FEATURES_BASE].copy()
X_test     = X_test_all[FEATURES_BASE].copy()
X_train_cb = X_train_all[FEATURES_CB].copy()
X_test_cb  = X_test_all[FEATURES_CB].copy()

for col in CAT_COLS_CB:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col]  = X_test_cb[col].astype(str)

print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"Delayed in test set: {y_test.sum():,} ({y_test.mean():.1%})")


# =============================================================================
# STEP 4B: TARGET ENCODING (XGB / LGBM / LR / RF)
# =============================================================================

def target_encode(train_df, test_df, col, target_col, smoothing=20):
    global_mean = train_df[target_col].mean()
    stats  = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
             (stats["count"] + smoothing)
    return (
        train_df[col].map(smooth).fillna(global_mean),
        test_df[col].map(smooth).fillna(global_mean)
    )

train_with_target = X_train.copy()
train_with_target[TARGET] = y_train.values

ENCODE_SETTINGS = {
    "dest_enc":              20,
    "carrier_enc":           20,
    "route_enc":             50,
    "carrier_origin_enc":    50,
    "carrier_month_enc":     30,   # medium smoothing — monthly cells have decent n
    "route_dep_period_enc":  80,   # high smoothing — very sparse combinations
}

for col, smoothing in ENCODE_SETTINGS.items():
    new_col = col.replace("_enc", "_delay_rate")
    X_train[new_col], X_test[new_col] = target_encode(
        train_with_target, X_test, col, TARGET, smoothing=smoothing
    )

X_train = X_train.drop(columns=list(ENCODE_SETTINGS.keys()))
X_test  = X_test.drop(columns=list(ENCODE_SETTINGS.keys()))
FEATURES_FINAL = list(X_train.columns)

print(f"\nXGB/LGBM features ({len(FEATURES_FINAL)} total): {FEATURES_FINAL}")
print(f"CatBoost features ({len(FEATURES_CB)} total, {len(CAT_COLS_CB)} raw cats)")


# =============================================================================
# UTILITY: THRESHOLD TUNING + EVAL
# =============================================================================

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall), 0
    )
    return float(thresholds[np.argmax(f1[:-1])]), float(np.max(f1[:-1]))


def evaluate(name, y_true, y_proba, verbose=True):
    thresh, _ = find_best_threshold(y_true, y_proba)
    preds = (y_proba >= thresh).astype(int)
    metrics = {
        "Accuracy":  accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall":    recall_score(y_true, preds),
        "F1 Score":  f1_score(y_true, preds),
        "AUC-ROC":   roc_auc_score(y_true, y_proba),
    }
    if verbose:
        print(f"  → Best threshold (max-F1): {thresh:.3f}")
        print(pd.Series(metrics).round(4).to_string())
        print("\n", classification_report(y_true, preds,
              target_names=["On Time (0)", "Delayed (1)"], zero_division=0))
    return metrics, thresh, preds


# =============================================================================
# STEP 5A: LOGISTIC REGRESSION
# =============================================================================
print("\n" + "="*50 + "\nLOGISTIC REGRESSION\n" + "="*50)
lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_metrics, lr_thresh, lr_preds = evaluate("LR", y_test, lr_proba)
coef_df = pd.DataFrame({"Feature": FEATURES_FINAL, "Coefficient": lr.coef_[0]}
                       ).sort_values("Coefficient", ascending=False)
print("\nCoefficients:")
print(coef_df.to_string(index=False))


# =============================================================================
# STEP 5B: RANDOM FOREST
# =============================================================================
print("\n" + "="*50 + "\nRANDOM FOREST\n" + "="*50)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_metrics, rf_thresh, rf_preds = evaluate("RF", y_test, rf_proba)


# =============================================================================
# STEP 5C: XGBOOST
# =============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*50 + "\nXGBOOST\n" + "="*50)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")
    xgb_model = XGBClassifier(
        n_estimators=5000, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, random_state=42, n_jobs=-1,
        tree_method="hist",   # histogram-based: ~5-10x less RAM than "exact"
        eval_metric="auc", early_stopping_rounds=50, verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    print(f"Best iteration: {xgb_model.best_iteration}")
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics, xgb_thresh, xgb_preds = evaluate("XGB", y_test, xgb_proba)
else:
    xgb_metrics = xgb_preds = xgb_proba = xgb_model = None


# =============================================================================
# STEP 5D: LIGHTGBM
# =============================================================================
if LGBM_AVAILABLE:
    print("\n" + "="*50 + "\nLIGHTGBM\n" + "="*50)
    lgbm_model = LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=7, num_leaves=63,
        class_weight="balanced",
        subsample=0.8, subsample_freq=5,
        colsample_bytree=0.7, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_metrics, lgbm_thresh, lgbm_preds = evaluate("LGBM", y_test, lgbm_proba)
else:
    lgbm_metrics = lgbm_preds = lgbm_proba = lgbm_model = None


# =============================================================================
# STEP 5E: CATBOOST — Optuna hyperparameter tuning
#
# CatBoost is the performance leader so we spend trials on it.
# Optuna samples depth, learning_rate, l2_leaf_reg, colsample_bylevel
# using TPE (Tree-structured Parzen Estimator) — smarter than grid search.
# Each trial runs a shortened CatBoost (500 iter) on a 20% subsample of
# the training data for speed. The best params are then used for the full
# training run.
# =============================================================================
if CATBOOST_AVAILABLE:
    print("\n" + "="*50 + "\nCATBOOST\n" + "="*50)

    cat_feature_indices = [FEATURES_CB.index(c) for c in CAT_COLS_CB]

    # --- Optuna tuning ---
    BEST_CB_PARAMS = {}

    if OPTUNA_AVAILABLE:
        print("Running Optuna (30 trials × 500 iter on 20% subsample)...")

        # Use a 20% subsample for speed during tuning
        idx = np.random.RandomState(0).choice(len(X_train_cb), size=int(0.2*len(X_train_cb)),
                                              replace=False)
        X_tune = X_train_cb.iloc[idx].reset_index(drop=True)
        y_tune = y_train.iloc[idx].reset_index(drop=True)

        # Inner validation split for Optuna
        X_t, X_v, y_t, y_v = train_test_split(X_tune, y_tune,
                                               test_size=0.2, random_state=7,
                                               stratify=y_tune)
        for col in CAT_COLS_CB:
            X_t[col] = X_t[col].astype(str)
            X_v[col] = X_v[col].astype(str)

        def objective(trial):
            params = {
                "iterations":        500,
                "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "depth":             trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 10, 50),
                "subsample":         0.8,
                "loss_function":     "Logloss",
                "eval_metric":       "AUC",
                "auto_class_weights": "Balanced",
                "random_seed":       42,
                "thread_count":      -1,
                "verbose":           False,
            }
            m = CatBoostClassifier(**params)
            m.fit(X_t, y_t, cat_features=cat_feature_indices,
                  eval_set=(X_v, y_v), use_best_model=True)
            return roc_auc_score(y_v, m.predict_proba(X_v)[:, 1])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        BEST_CB_PARAMS = study.best_params
        print(f"\nBest Optuna AUC (500-iter, 20% subsample): {study.best_value:.4f}")
        print(f"Best params: {BEST_CB_PARAMS}")
    else:
        # Defaults used in v8/v9 — still strong
        BEST_CB_PARAMS = {
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "colsample_bylevel": 0.7,
            "min_data_in_leaf": 20,
        }

    # --- Full CatBoost training with best params ---
    print("\nTraining CatBoost with best params (5000 iter, full 1M)...")
    cb_model = CatBoostClassifier(
        iterations=5000,
        learning_rate=BEST_CB_PARAMS.get("learning_rate", 0.05),
        depth=BEST_CB_PARAMS.get("depth", 6),
        l2_leaf_reg=BEST_CB_PARAMS.get("l2_leaf_reg", 3.0),
        colsample_bylevel=BEST_CB_PARAMS.get("colsample_bylevel", 0.7),
        min_data_in_leaf=BEST_CB_PARAMS.get("min_data_in_leaf", 20),
        subsample=0.8,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=-1,
        verbose=100,
        early_stopping_rounds=50,
        train_dir="outputs/catboost_info",
    )
    cb_model.fit(
        X_train_cb, y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test_cb, y_test),
        use_best_model=True,
    )
    print(f"Best iteration: {cb_model.best_iteration_}")
    cb_proba = cb_model.predict_proba(X_test_cb)[:, 1]
    cb_metrics, cb_thresh, cb_preds = evaluate("CatBoost", y_test, cb_proba)
else:
    cb_metrics = cb_preds = cb_proba = cb_model = None


# =============================================================================
# STEP 5F: 3-WAY ENSEMBLE — simple blends
# =============================================================================
print("\n" + "="*50 + "\n3-WAY ENSEMBLE\n" + "="*50)

have_all_three = XGBOOST_AVAILABLE and LGBM_AVAILABLE and CATBOOST_AVAILABLE
have_xgb_cb   = XGBOOST_AVAILABLE and CATBOOST_AVAILABLE

if have_all_three:
    # Equal 3-way
    ens3_equal = (xgb_proba + lgbm_proba + cb_proba) / 3
    print("--- Equal 3-way (XGB + LGBM + CB) ---")
    ens3_eq_metrics, _, ens3_eq_preds = evaluate("Ensemble 3-way equal", y_test, ens3_equal)

    # AUC-weighted 3-way
    aucs = np.array([xgb_metrics["AUC-ROC"], lgbm_metrics["AUC-ROC"], cb_metrics["AUC-ROC"]])
    ws   = aucs / aucs.sum()
    print(f"\n--- Weighted 3-way (XGB×{ws[0]:.2f} + LGBM×{ws[1]:.2f} + CB×{ws[2]:.2f}) ---")
    ens3_wt = ws[0]*xgb_proba + ws[1]*lgbm_proba + ws[2]*cb_proba
    ens3_wt_metrics, _, ens3_wt_preds = evaluate("Ensemble 3-way weighted", y_test, ens3_wt)
else:
    ens3_eq_metrics = ens3_wt_metrics = None


# =============================================================================
# STEP 5G: STACKING META-LEARNER
#
# Instead of guessing weights, train a Logistic Regression on the out-of-fold
# (OOF) probability predictions of the 3 base models. This learns the optimal
# combination from data. Uses 5-fold stratified CV on the training set.
#
# The meta-features are [xgb_oof, lgbm_oof, cb_oof] for each training sample.
# The meta-learner is then applied to base model predictions on the test set.
#
# Note: CatBoost OOF predictions require fitting on each fold's training subset,
# which is slow (5 × ~650k rows each). We use 300-iter CatBoost for speed.
# =============================================================================
if have_all_three:
    print("\n" + "="*50 + "\nSTACKING META-LEARNER (5-fold OOF)\n" + "="*50)
    print("This will take ~10-15 minutes (5 folds × 3 models)...")

    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Align indices — X_train is indexed on a subset of model_df
    X_tr_np   = X_train.values
    X_tr_cb_np = X_train_cb.values   # keep as DataFrame for cat_features
    y_tr_np   = y_train.values

    oof_xgb  = np.zeros(len(X_train))
    oof_lgbm = np.zeros(len(X_train))
    oof_cb   = np.zeros(len(X_train))

    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_np, y_tr_np)):
        print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ", flush=True)

        X_f_tr, X_f_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_f_tr, y_f_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        X_f_cb_tr  = X_train_cb.iloc[tr_idx].copy()
        X_f_cb_val = X_train_cb.iloc[val_idx].copy()
        for col in CAT_COLS_CB:
            X_f_cb_tr[col]  = X_f_cb_tr[col].astype(str)
            X_f_cb_val[col] = X_f_cb_val[col].astype(str)

        # XGBoost fold
        xgb_f = XGBClassifier(
            n_estimators=xgb_model.best_iteration,  # use best iter from full run
            max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pw, subsample=0.8, colsample_bytree=0.7,
            min_child_weight=10, random_state=42, n_jobs=-1,
            tree_method="hist",
            eval_metric="auc", verbosity=0
        )
        xgb_f.fit(X_f_tr, y_f_tr)
        oof_xgb[val_idx] = xgb_f.predict_proba(X_f_val)[:, 1]

        # LightGBM fold
        lgbm_f = LGBMClassifier(
            n_estimators=lgbm_model.best_iteration_,
            learning_rate=0.05, max_depth=7, num_leaves=63,
            class_weight="balanced", subsample=0.8, subsample_freq=5,
            colsample_bytree=0.7, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgbm_f.fit(X_f_tr, y_f_tr)
        oof_lgbm[val_idx] = lgbm_f.predict_proba(X_f_val)[:, 1]

        # CatBoost fold (300 iter for speed)
        cb_f = CatBoostClassifier(
            iterations=300,
            learning_rate=BEST_CB_PARAMS.get("learning_rate", 0.05),
            depth=BEST_CB_PARAMS.get("depth", 6),
            l2_leaf_reg=BEST_CB_PARAMS.get("l2_leaf_reg", 3.0),
            subsample=0.8, auto_class_weights="Balanced",
            random_seed=42, thread_count=-1, verbose=False
        )
        cb_f.fit(X_f_cb_tr, y_f_tr, cat_features=cat_feature_indices)
        oof_cb[val_idx] = cb_f.predict_proba(X_f_cb_val)[:, 1]

        print(f"done")

    # Train meta-learner on OOF predictions
    meta_X_train = np.column_stack([oof_xgb, oof_lgbm, oof_cb])
    meta_X_test  = np.column_stack([xgb_proba, lgbm_proba, cb_proba])

    meta_lr = LogisticRegression(max_iter=500, random_state=42)
    meta_lr.fit(meta_X_train, y_tr_np)

    stack_proba = meta_lr.predict_proba(meta_X_test)[:, 1]
    print(f"\nMeta-learner weights: XGB={meta_lr.coef_[0][0]:.3f}  "
          f"LGBM={meta_lr.coef_[0][1]:.3f}  CB={meta_lr.coef_[0][2]:.3f}")
    stack_metrics, stack_thresh, stack_preds = evaluate("Stacking", y_test, stack_proba)
else:
    stack_metrics = stack_preds = stack_proba = None


# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================

model_names        = ["Logistic Regression", "Random Forest"]
model_preds        = [lr_preds, rf_preds]
model_probas       = [lr_proba, rf_proba]
model_metrics_list = [lr_metrics, rf_metrics]

for flag, name, preds, proba, metrics in [
    (XGBOOST_AVAILABLE,             "XGBoost",            xgb_preds,     xgb_proba,  xgb_metrics),
    (LGBM_AVAILABLE,                "LightGBM",           lgbm_preds,    lgbm_proba, lgbm_metrics),
    (CATBOOST_AVAILABLE,            "CatBoost",           cb_preds,      cb_proba,   cb_metrics),
    (have_all_three,                "Ensemble 3-way",     ens3_wt_preds, ens3_wt,    ens3_wt_metrics),
    (have_all_three,                "Stacking",           stack_preds,   stack_proba,stack_metrics),
]:
    if flag and preds is not None:
        model_names.append(name)
        model_preds.append(preds)
        model_probas.append(proba)
        model_metrics_list.append(metrics)

n_models = len(model_names)

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]
for ax, preds, title in zip(axes, model_preds, model_names):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["On Time", "Delayed"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title, fontsize=8, fontweight="bold")
plt.suptitle("Confusion Matrices (threshold-tuned)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- ROC Curves ---
colors = ["tab:blue","tab:orange","tab:green","tab:red",
          "tab:purple","black","tab:brown"]
linestyles = ["-","-","-","-","-","--","--"]
fig, ax = plt.subplots(figsize=(10, 7))
for name, proba, metrics, color, ls in zip(
    model_names, model_probas, model_metrics_list, colors, linestyles
):
    RocCurveDisplay.from_predictions(
        y_test, proba,
        name=f"{name} (AUC={metrics['AUC-ROC']:.3f})",
        ax=ax, color=color, linestyle=ls
    )
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Baseline")
ax.set_title("ROC Curves — Flight Delay Classification", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Feature importances ---
for imp, name, color, fname in [
    (pd.Series(rf.feature_importances_, index=FEATURES_FINAL), "Random Forest", "steelblue", "charts/feature_importance_rf.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.sort_values(ascending=True).plot(kind="barh", ax=ax, color=color)
    ax.set_title(f"{name} — Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

if XGBOOST_AVAILABLE:
    imp = pd.Series(xgb_model.feature_importances_, index=FEATURES_FINAL)
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.sort_values(ascending=True).plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_xgb.png", dpi=150, bbox_inches="tight")
    plt.show()

if CATBOOST_AVAILABLE:
    imp = pd.Series(cb_model.get_feature_importance(), index=FEATURES_CB)
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.sort_values(ascending=True).plot(kind="barh", ax=ax, color="mediumpurple")
    ax.set_title("CatBoost — Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_catboost.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- Summary table ---
comparison = pd.DataFrame(
    {n: m for n, m in zip(model_names, model_metrics_list)}
).T
print("\n=== Exploration Run Results (1M rows, threshold-tuned) ===")
print(comparison.round(4).to_string())

fig, ax = plt.subplots(figsize=(14, 4))
comparison.plot(kind="bar", ax=ax, ylim=(0, 1), colormap="Set2", edgecolor="black")
ax.set_title("Model Comparison (1M sample, threshold-tuned)", fontsize=13, fontweight="bold")
ax.set_xticklabels(comparison.index, rotation=20, ha="right")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# STEP 7: FULL 6.4M FINAL RUN
# NOTE: ~45-90 minutes. Comment out if not needed.
# =============================================================================
print("\n" + "="*60)
print("STEP 7: FULL DATA RUN (6.4M rows)")
print("="*60)

# Re-query the DB here — avoids holding two large DataFrames in RAM
# simultaneously during the exploration run (was causing XGBoost OOM).
conn_full = sqlite3.connect(DB_PATH)
df_full = pd.read_sql_query(query, conn_full)
conn_full.close()
float_cols_f = df_full.select_dtypes(include="float64").columns
df_full[float_cols_f] = df_full[float_cols_f].astype("float32")
print(f"Reloaded {len(df_full):,} rows for full run")

df_full = engineer_features(df_full)
df_full, _ = encode_categoricals(df_full)

full_all_cols = list(dict.fromkeys(FEATURES_BASE + FEATURES_CB + [TARGET]))
full_model_df = df_full[full_all_cols].dropna()
X_full_all = full_model_df.drop(columns=[TARGET])
y_full     = full_model_df[TARGET].astype(int)

X_train_f_all, X_test_f_all, y_train_f, y_test_f = train_test_split(
    X_full_all, y_full, test_size=0.2, random_state=42, stratify=y_full
)
print(f"Full run — Train: {len(X_train_f_all):,}  |  Test: {len(X_test_f_all):,}")

X_train_f    = X_train_f_all[FEATURES_BASE].copy()
X_test_f     = X_test_f_all[FEATURES_BASE].copy()
X_train_f_cb = X_train_f_all[FEATURES_CB].copy()
X_test_f_cb  = X_test_f_all[FEATURES_CB].copy()
for col in CAT_COLS_CB:
    X_train_f_cb[col] = X_train_f_cb[col].astype(str)
    X_test_f_cb[col]  = X_test_f_cb[col].astype(str)

train_full_target = X_train_f.copy()
train_full_target[TARGET] = y_train_f.values
for col, smoothing in ENCODE_SETTINGS.items():
    new_col = col.replace("_enc", "_delay_rate")
    X_train_f[new_col], X_test_f[new_col] = target_encode(
        train_full_target, X_test_f, col, TARGET, smoothing=smoothing
    )
X_train_f = X_train_f.drop(columns=list(ENCODE_SETTINGS.keys()))
X_test_f  = X_test_f.drop(columns=list(ENCODE_SETTINGS.keys()))

full_probas = {}

if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost on full 6.4M rows...")
    scale_pw_f = (y_train_f == 0).sum() / (y_train_f == 1).sum()
    xgb_full = XGBClassifier(
        n_estimators=5000, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pw_f, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, random_state=42, n_jobs=-1,
        tree_method="hist",   # histogram-based: ~5-10x less RAM than "exact"
        eval_metric="auc", early_stopping_rounds=50, verbosity=0
    )
    xgb_full.fit(X_train_f, y_train_f, eval_set=[(X_test_f, y_test_f)], verbose=100)
    xgb_full_proba = xgb_full.predict_proba(X_test_f)[:, 1]
    xgb_full_metrics, _, _ = evaluate("XGBoost Full", y_test_f, xgb_full_proba)
    full_probas["XGBoost"] = (xgb_full_proba, xgb_full_metrics)
    print(f"XGBoost FULL — AUC: {xgb_full_metrics['AUC-ROC']:.4f}  F1: {xgb_full_metrics['F1 Score']:.4f}")

if LGBM_AVAILABLE:
    print("\nTraining LightGBM on full 6.4M rows...")
    lgbm_full = LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=7, num_leaves=63,
        class_weight="balanced", subsample=0.8, subsample_freq=5,
        colsample_bytree=0.7, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_full.fit(
        X_train_f, y_train_f, eval_set=[(X_test_f, y_test_f)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]
    )
    lgbm_full_proba = lgbm_full.predict_proba(X_test_f)[:, 1]
    lgbm_full_metrics, _, _ = evaluate("LightGBM Full", y_test_f, lgbm_full_proba)
    full_probas["LightGBM"] = (lgbm_full_proba, lgbm_full_metrics)
    print(f"LightGBM FULL — AUC: {lgbm_full_metrics['AUC-ROC']:.4f}  F1: {lgbm_full_metrics['F1 Score']:.4f}")

if CATBOOST_AVAILABLE:
    print("\nTraining CatBoost on full 6.4M rows (Optuna params)...")
    cb_full = CatBoostClassifier(
        iterations=5000,
        learning_rate=BEST_CB_PARAMS.get("learning_rate", 0.05),
        depth=BEST_CB_PARAMS.get("depth", 6),
        l2_leaf_reg=BEST_CB_PARAMS.get("l2_leaf_reg", 3.0),
        colsample_bylevel=BEST_CB_PARAMS.get("colsample_bylevel", 0.7),
        min_data_in_leaf=BEST_CB_PARAMS.get("min_data_in_leaf", 20),
        subsample=0.8, loss_function="Logloss", eval_metric="AUC",
        auto_class_weights="Balanced", random_seed=42, thread_count=-1,
        verbose=100, early_stopping_rounds=50,
        train_dir="outputs/catboost_info",
    )
    cb_full.fit(
        X_train_f_cb, y_train_f,
        cat_features=cat_feature_indices,
        eval_set=(X_test_f_cb, y_test_f),
        use_best_model=True,
    )
    cb_full_proba = cb_full.predict_proba(X_test_f_cb)[:, 1]
    cb_full_metrics, _, _ = evaluate("CatBoost Full", y_test_f, cb_full_proba)
    full_probas["CatBoost"] = (cb_full_proba, cb_full_metrics)
    print(f"CatBoost FULL — AUC: {cb_full_metrics['AUC-ROC']:.4f}  F1: {cb_full_metrics['F1 Score']:.4f}")

# --- Full ensemble ---
if len(full_probas) == 3:
    all_p = np.array([v[0] for v in full_probas.values()])
    all_a = np.array([v[1]["AUC-ROC"] for v in full_probas.values()])
    ws = all_a / all_a.sum()
    ens_full = (ws[:, None] * all_p).sum(axis=0)
    ens_full_metrics, _, _ = evaluate("Ensemble Full", y_test_f, ens_full)
    print(f"\nEnsemble FULL  — AUC: {ens_full_metrics['AUC-ROC']:.4f}  F1: {ens_full_metrics['F1 Score']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"XGBoost": "tab:green", "LightGBM": "tab:orange", "CatBoost": "tab:purple"}
    for mname, (proba, metrics) in full_probas.items():
        RocCurveDisplay.from_predictions(
            y_test_f, proba,
            name=f"{mname} 6.4M (AUC={metrics['AUC-ROC']:.3f})",
            ax=ax, color=palette[mname]
        )
    RocCurveDisplay.from_predictions(
        y_test_f, ens_full,
        name=f"Ensemble 6.4M (AUC={ens_full_metrics['AUC-ROC']:.3f})",
        ax=ax, color="tab:red", linestyle="--"
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("ROC Curves — Full 6.4M Run", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("charts/roc_curves_full_run.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nAll charts saved.")


# =============================================================================
# NOTES
# =============================================================================
# 1. New features:
#    carrier_month: carrier × month captures seasonal performance variation
#      (some carriers are much worse in August or December specifically)
#    route_dep_period: route × time-of-day block. Smoothing=80 because
#      most route×period cells have relatively few observations.
#    is_early_morning: pre-6am flights are first-of-day; different delay
#      profile (no upstream cascading, but maintenance issues surface here)
#
# 2. 3-way ensemble: LightGBM (leaf-wise) + XGBoost (depth-wise) +
#    CatBoost (oblivious trees) represent genuinely different inductive
#    biases — more likely to have uncorrelated errors than XGB+CB alone
#
# 3. Stacking: OOF meta-learner learns the optimal blend from data.
#    The LR coefficient on each base model tells you how much the meta-
#    learner trusts it relative to the others.
#
# 4. Optuna: 30 trials × 500 iter on 20% subsample. ~5-10 minutes.
#    Install: pip install optuna
#
# 5. Full run is the single biggest remaining gain — 6.4M rows gives
#    target encodings 6× more stable and CatBoost's internal statistics
#    cover rare routes that were missing in the 1M sample.
# =============================================================================
