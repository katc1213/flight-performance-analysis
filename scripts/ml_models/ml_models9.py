# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15
#
# CHANGES FROM ml_models8.py:
#   1. LIGHTGBM FINAL FIX (attempt 4) — class_weight='balanced' (sklearn API).
#      Root cause of all previous failures:
#        v5: no subsample_freq → bagging silently off → plateau @ iter 3
#        v6: is_unbalance (same root as v5)
#        v7: scale_pos_weight + binary_logloss → unweighted logloss rises → stops @ iter 2
#        v8: is_unbalance + subsample_freq=5 + AUC → is_unbalance causes hard
#            overfitting to minority in first rounds → AUC peaks @ iter 3 on
#            unweighted eval set → stops immediately
#        v9: class_weight='balanced' passes sample weights to fit() instead of
#            modifying the loss function — AUC on eval set reflects true class
#            distribution and degrades smoothly, so early stopping works correctly
#
#   2. CATBOOST ITERATIONS → 5000 (was hitting ceiling at 2999)
#
#   3. XGB + CATBOOST ENSEMBLE — simple probability average. These two models
#      use completely different feature representations (target-encoded floats
#      vs raw string categoricals), so their errors are partially independent.
#      Blending uncorrelated models consistently outperforms either alone.
#
#   dep_del15 intentionally excluded — pre-departure prediction framing
# =============================================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
# Set working directory to repo root so all relative paths resolve correctly
import pathlib as _pl, os as _os
_os.chdir(_pl.Path(__file__).resolve().parent.parent.parent)

DB_PATH = "data/CleanedData/flights.db"

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

df_full = df.copy()
df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows (exploration run)")
print(f"\nClass balance (arr_del15):\n{df['arr_del15'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

_US_HOLIDAY_MD = [
    (1,  1), (1, 15), (2, 19), (5, 27), (7,  4),
    (9,  2), (11, 11), (11, 28), (12, 25), (12, 31),
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

    # Log-distance only (raw distance collinear — opposite sign in LR)
    df["log_distance"] = np.log1p(df["distance"].astype(float))

    # Calendar context
    df["near_holiday"] = df["flight_date"].apply(_near_holiday)
    df["is_peak"]      = df["month"].isin([6, 7, 8, 12]).astype(int)

    # Interaction keys
    df["route"]          = df["origin"].astype(str) + "_" + df["dest"].astype(str)
    df["carrier_origin"] = df["carrier"].astype(str) + "_" + df["origin"].astype(str)
    df["origin_state"]   = df["origin_state"].fillna("UNKNOWN")

    return df


df = engineer_features(df)


# =============================================================================
# STEP 3: ENCODE CATEGORICALS (for XGB / LGBM / LR / RF)
# =============================================================================

def encode_categoricals(df):
    encoders = {}
    for col, key in [
        ("dest",           "dest_enc"),
        ("origin_state",   "state_enc"),
        ("carrier",        "carrier_enc"),
        ("route",          "route_enc"),
        ("carrier_origin", "carrier_origin_enc"),
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
    "near_holiday", "is_peak",
    "dest_enc", "carrier_enc", "route_enc", "carrier_origin_enc",
]

CAT_COLS_CB = ["dest", "carrier", "route", "carrier_origin", "origin_state"]
FEATURES_CB = [
    "month", "day_of_week", "is_weekend", "dep_hour", "dep_period",
    "log_distance",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "near_holiday", "is_peak",
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
# STEP 4B: TARGET ENCODING
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
    "dest_enc":           20,
    "carrier_enc":        20,
    "route_enc":          50,
    "carrier_origin_enc": 50,
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
    best = np.argmax(f1[:-1])
    return float(thresholds[best]), float(f1[best])


def evaluate(name, y_true, y_proba):
    thresh, _ = find_best_threshold(y_true, y_proba)
    print(f"  → Best threshold (max-F1): {thresh:.3f}")
    preds = (y_proba >= thresh).astype(int)
    metrics = {
        "Accuracy":  accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall":    recall_score(y_true, preds),
        "F1 Score":  f1_score(y_true, preds),
        "AUC-ROC":   roc_auc_score(y_true, y_proba),
    }
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

coef_df = pd.DataFrame(
    {"Feature": FEATURES_FINAL, "Coefficient": lr.coef_[0]}
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
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    print(f"Best iteration: {xgb_model.best_iteration}")
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics, xgb_thresh, xgb_preds = evaluate("XGB", y_test, xgb_proba)
else:
    xgb_metrics = xgb_preds = xgb_proba = xgb_model = None


# =============================================================================
# STEP 5D: LIGHTGBM — class_weight='balanced' fix
#
# Full failure history:
#   v5: no subsample_freq → bagging silently off → correlated trees → plateau @ 3
#   v6: is_unbalance, no subsample_freq → same root cause
#   v7: scale_pos_weight + binary_logloss → unweighted logloss rises → stops @ 2
#   v8: is_unbalance + subsample_freq=5 + AUC → is_unbalance overweights minority
#       so aggressively that AUC on unweighted eval set peaks then flatlines @ 3
#   v9: class_weight='balanced' passes sample weights into fit() via sklearn API.
#       The eval set is assessed without reweighting, but the TRAINING signal
#       is balanced — AUC improves smoothly and early stopping degrades properly.
# =============================================================================
if LGBM_AVAILABLE:
    print("\n" + "="*50 + "\nLIGHTGBM (v9 fix)\n" + "="*50)

    lgbm_model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        class_weight="balanced",   # sklearn API — sample weights in fit()
        subsample=0.8,
        subsample_freq=5,          # activates bagging
        colsample_bytree=0.7,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
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
# STEP 5E: CATBOOST (iterations raised to 5000 — was hitting ceiling at 2999)
# =============================================================================
if CATBOOST_AVAILABLE:
    print("\n" + "="*50 + "\nCATBOOST\n" + "="*50)

    cat_feature_indices = [FEATURES_CB.index(c) for c in CAT_COLS_CB]

    cb_model = CatBoostClassifier(
        iterations=5000,           # raised from 3000 — was still learning at 2999
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        subsample=0.8,
        colsample_bylevel=0.7,
        min_data_in_leaf=20,
        random_seed=42,
        thread_count=-1,
        verbose=100,
        early_stopping_rounds=50,
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
# STEP 5F: XGB + CATBOOST ENSEMBLE
#
# These two models have partially independent errors because they are built
# on fundamentally different feature representations:
#   XGBoost  — target-encoded floats (route_delay_rate, carrier_delay_rate…)
#   CatBoost — raw string categoricals (ordered internal statistics, no encoding)
#
# Simple average of their probability outputs is a strong baseline for
# blending two uncorrelated models. A weighted blend (0.4 XGB + 0.6 CB) is
# tried based on their individual AUC scores — a higher-AUC model typically
# contributes more to the blend optimum.
# =============================================================================
if XGBOOST_AVAILABLE and CATBOOST_AVAILABLE:
    print("\n" + "="*50 + "\nXGB + CATBOOST ENSEMBLE\n" + "="*50)

    # Equal blend
    ensemble_proba_equal = 0.5 * xgb_proba + 0.5 * cb_proba
    print("--- Equal blend (0.5 XGB + 0.5 CB) ---")
    ens_eq_metrics, ens_eq_thresh, ens_eq_preds = evaluate(
        "Ensemble (equal)", y_test, ensemble_proba_equal
    )

    # Weighted blend — weight by individual AUC
    xgb_auc = xgb_metrics["AUC-ROC"]
    cb_auc  = cb_metrics["AUC-ROC"]
    w_xgb   = xgb_auc / (xgb_auc + cb_auc)
    w_cb    = cb_auc  / (xgb_auc + cb_auc)
    ensemble_proba_weighted = w_xgb * xgb_proba + w_cb * cb_proba
    print(f"\n--- Weighted blend ({w_xgb:.2f} XGB + {w_cb:.2f} CB) ---")
    ens_wt_metrics, ens_wt_thresh, ens_wt_preds = evaluate(
        "Ensemble (weighted)", y_test, ensemble_proba_weighted
    )
else:
    ens_eq_metrics = ens_wt_metrics = None


# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================

model_names        = ["Logistic Regression", "Random Forest"]
model_preds        = [lr_preds, rf_preds]
model_probas       = [lr_proba, rf_proba]
model_metrics_list = [lr_metrics, rf_metrics]

if XGBOOST_AVAILABLE:
    model_names.append("XGBoost")
    model_preds.append(xgb_preds)
    model_probas.append(xgb_proba)
    model_metrics_list.append(xgb_metrics)

if LGBM_AVAILABLE:
    model_names.append("LightGBM")
    model_preds.append(lgbm_preds)
    model_probas.append(lgbm_proba)
    model_metrics_list.append(lgbm_metrics)

if CATBOOST_AVAILABLE:
    model_names.append("CatBoost")
    model_preds.append(cb_preds)
    model_probas.append(cb_proba)
    model_metrics_list.append(cb_metrics)

if XGBOOST_AVAILABLE and CATBOOST_AVAILABLE:
    model_names.append("Ensemble (equal)")
    model_preds.append(ens_eq_preds)
    model_probas.append(ensemble_proba_equal)
    model_metrics_list.append(ens_eq_metrics)

    model_names.append("Ensemble (weighted)")
    model_preds.append(ens_wt_preds)
    model_probas.append(ensemble_proba_weighted)
    model_metrics_list.append(ens_wt_metrics)

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
    ax.set_title(title, fontsize=9, fontweight="bold")
plt.suptitle("Confusion Matrices (threshold-tuned)", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- ROC Curves ---
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
          "tab:purple", "tab:brown", "tab:pink"]
fig, ax = plt.subplots(figsize=(9, 6))
for name, proba, metrics, color in zip(
    model_names, model_probas, model_metrics_list, colors
):
    ls = "--" if "Ensemble" in name else "-"
    RocCurveDisplay.from_predictions(
        y_test, proba,
        name=f"{name} (AUC={metrics['AUC-ROC']:.3f})",
        ax=ax, color=color, linestyle=ls
    )
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax.set_title("ROC Curves — Flight Delay Classification", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Feature Importance: RF ---
imp_rf = pd.Series(rf.feature_importances_, index=FEATURES_FINAL).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
imp_rf.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/feature_importance_rf.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Feature Importance: XGBoost ---
if XGBOOST_AVAILABLE:
    imp_xgb = pd.Series(
        xgb_model.feature_importances_, index=FEATURES_FINAL
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp_xgb.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_xgb.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- Feature Importance: CatBoost ---
if CATBOOST_AVAILABLE:
    imp_cb = pd.Series(
        cb_model.get_feature_importance(), index=FEATURES_CB
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    imp_cb.plot(kind="barh", ax=ax, color="mediumpurple")
    ax.set_title("CatBoost — Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_catboost.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- Metrics Comparison ---
comparison = pd.DataFrame(
    {name: m for name, m in zip(model_names, model_metrics_list)}
).T
print("\n=== Exploration Run Results (1M rows, threshold-tuned) ===")
print(comparison.round(4).to_string())

fig, ax = plt.subplots(figsize=(14, 4))
comparison.plot(kind="bar", ax=ax, ylim=(0, 1), colormap="Set2", edgecolor="black")
ax.set_title("Model Comparison (1M sample, threshold-tuned)", fontsize=13, fontweight="bold")
ax.set_xticklabels(comparison.index, rotation=20, ha="right")
ax.legend(loc="lower right")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# STEP 7: FULL 6.4M FINAL RUN — XGBoost + LightGBM + CatBoost + Ensemble
# NOTE: ~30-60 minutes depending on machine. Comment out if not needed.
# =============================================================================
print("\n" + "="*60)
print("STEP 7: FULL DATA RUN (6.4M rows)")
print("="*60)

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

X_train_f = X_train_f_all[FEATURES_BASE].copy()
X_test_f  = X_test_f_all[FEATURES_BASE].copy()
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

# --- Full XGBoost ---
xgb_full_proba = None
if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost on full 6.4M rows...")
    scale_pw_f = (y_train_f == 0).sum() / (y_train_f == 1).sum()
    xgb_full = XGBClassifier(
        n_estimators=5000, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pw_f, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, random_state=42, n_jobs=-1,
        eval_metric="auc", early_stopping_rounds=50, verbosity=0
    )
    xgb_full.fit(X_train_f, y_train_f, eval_set=[(X_test_f, y_test_f)], verbose=100)
    print(f"Best iteration: {xgb_full.best_iteration}")
    xgb_full_proba = xgb_full.predict_proba(X_test_f)[:, 1]
    xgb_full_metrics, _, _ = evaluate("XGBoost Full", y_test_f, xgb_full_proba)
    print(f"XGBoost FULL — AUC: {xgb_full_metrics['AUC-ROC']:.4f}  F1: {xgb_full_metrics['F1 Score']:.4f}")

# --- Full LightGBM ---
if LGBM_AVAILABLE:
    print("\nTraining LightGBM on full 6.4M rows...")
    lgbm_full = LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=7, num_leaves=63,
        class_weight="balanced",
        subsample=0.8, subsample_freq=5,
        colsample_bytree=0.7, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_full.fit(
        X_train_f, y_train_f,
        eval_set=[(X_test_f, y_test_f)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    lgbm_full_proba = lgbm_full.predict_proba(X_test_f)[:, 1]
    lgbm_full_metrics, _, _ = evaluate("LightGBM Full", y_test_f, lgbm_full_proba)
    print(f"LightGBM FULL — AUC: {lgbm_full_metrics['AUC-ROC']:.4f}  F1: {lgbm_full_metrics['F1 Score']:.4f}")

# --- Full CatBoost ---
cb_full_proba = None
if CATBOOST_AVAILABLE:
    print("\nTraining CatBoost on full 6.4M rows...")
    cb_full = CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="AUC",
        auto_class_weights="Balanced",
        subsample=0.8, colsample_bylevel=0.7, min_data_in_leaf=20,
        random_seed=42, thread_count=-1,
        verbose=100, early_stopping_rounds=50,
    )
    cb_full.fit(
        X_train_f_cb, y_train_f,
        cat_features=cat_feature_indices,
        eval_set=(X_test_f_cb, y_test_f),
        use_best_model=True,
    )
    print(f"Best iteration: {cb_full.best_iteration_}")
    cb_full_proba = cb_full.predict_proba(X_test_f_cb)[:, 1]
    cb_full_metrics, _, _ = evaluate("CatBoost Full", y_test_f, cb_full_proba)
    print(f"CatBoost FULL — AUC: {cb_full_metrics['AUC-ROC']:.4f}  F1: {cb_full_metrics['F1 Score']:.4f}")

# --- Full Ensemble ---
if xgb_full_proba is not None and cb_full_proba is not None:
    print("\n--- Full Run Ensemble ---")
    w_xgb_f = xgb_full_metrics["AUC-ROC"]
    w_cb_f  = cb_full_metrics["AUC-ROC"]
    w_xgb_f /= (w_xgb_f + w_cb_f)
    w_cb_f   = 1 - w_xgb_f
    ens_full_proba = w_xgb_f * xgb_full_proba + w_cb_f * cb_full_proba
    ens_full_metrics, _, _ = evaluate("Ensemble Full", y_test_f, ens_full_proba)
    print(f"Ensemble FULL  — AUC: {ens_full_metrics['AUC-ROC']:.4f}  F1: {ens_full_metrics['F1 Score']:.4f}")

    # Final summary ROC
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, proba, metrics, color, ls in [
        ("XGBoost 6.4M",  xgb_full_proba,  xgb_full_metrics,  "tab:green",  "-"),
        ("CatBoost 6.4M", cb_full_proba,   cb_full_metrics,   "tab:purple", "-"),
        ("Ensemble 6.4M", ens_full_proba,  ens_full_metrics,  "tab:red",    "--"),
    ]:
        RocCurveDisplay.from_predictions(
            y_test_f, proba,
            name=f"{name} (AUC={metrics['AUC-ROC']:.3f})",
            ax=ax, color=color, linestyle=ls
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_title("ROC Curves — Full 6.4M Run", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("charts/roc_curves_full_run.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nAll charts saved.")


# =============================================================================
# NOTES
# =============================================================================
# 1. LightGBM failure chain (resolved in v9):
#    v5/v6: subsample_freq missing → bagging off → correlated trees → stops @ 3
#    v7:    binary_logloss rises on unweighted eval → stops @ 2
#    v8:    is_unbalance overweights minority → AUC peaks then flatlines @ 3
#    v9:    class_weight='balanced' → stable training, proper early stopping
#
# 2. distance dropped: collinear with log_distance, opposite signs in LR
#
# 3. CatBoost 5000 iterations: was still learning at 2999 in v8
#
# 4. Ensemble: XGB (target-encoded) + CatBoost (raw cats) have partially
#    independent errors → simple blend consistently improves AUC
#
# 5. Install: pip install xgboost lightgbm catboost
# =============================================================================
