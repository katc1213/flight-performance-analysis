# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15 (already a boolean in the flight table)
#
# CHANGES FROM PREVIOUS VERSION:
#   1. LightGBM fix — replaced scale_pos_weight with is_unbalance=True,
#      raised min_child_samples to 50 (prevents collapse at iteration 3)
#   2. XGBoost ceiling raised — n_estimators=3000, early_stopping_rounds=50
#   3. Smoothing tuned — route_enc and carrier_origin_enc use smoothing=50
#      (rare routes pulled harder toward global mean)
#   4. origin_delay_rate removed — route_delay_rate already contains this
#      signal; keeping both was creating negative interference (LR coef -0.51)
#   5. Full 6.4M row final training run for XGBoost and LightGBM after
#      the 1M exploration run completes
#
#   dep_del15 intentionally excluded — pre-departure prediction framing
#   (see project plan Section 5.2)
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
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
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

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: LOAD DATA FROM SQLITE
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

# Downcast float64 to float32 to halve RAM usage
float_cols = df.select_dtypes(include="float64").columns
df[float_cols] = df[float_cols].astype("float32")
print(f"Memory usage after downcast: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Sample to 1M rows for the exploration run.
# The full 6.4M run happens at the bottom of this file (STEP 7).
df_full = df.copy()   # keep a reference to the full dataset for the final run

df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows (exploration run)")

print(f"\nClass balance (arr_del15):\n{df['arr_del15'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    All feature engineering in one function so we can call it identically
    on both the 1M sample and the full 6.4M dataset.
    """
    df = df.copy()

    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df["month"]       = df["flight_date"].dt.month
    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["dep_hour"] = pd.to_datetime(
        df["crs_dep_time"], format="%H:%M:%S", errors="coerce"
    ).dt.hour
    # Fallback if SQLite stores time as integer HHMM:
    # df["dep_hour"] = df["crs_dep_time"].astype(float).floordiv(100).astype(int)

    df["dep_period"] = pd.cut(
        df["dep_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=[0, 1, 2, 3, 4]
    ).astype("Int64").fillna(2).astype(int)

    # Route pair and carrier+origin interaction
    df["route"]          = df["origin"].astype(str) + "_" + df["dest"].astype(str)
    df["carrier_origin"] = df["carrier"].astype(str) + "_" + df["origin"].astype(str)

    return df

df = engineer_features(df)

print("\nSample of engineered features:")
print(df[["flight_date", "month", "day_of_week", "is_weekend",
          "dep_hour", "dep_period", "distance", "route", "arr_del15"]].head())


# =============================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =============================================================================

def encode_categoricals(df):
    """Label-encode all categoricals. Returns df + fitted encoders dict."""
    encoders = {}
    for col, key in [
        ("dest",           "dest_enc"),
        ("origin_state",   "state_enc"),   # SQL aliases a.state AS origin_state
        ("carrier",        "carrier_enc"),
        ("route",          "route_enc"),
        ("carrier_origin", "carrier_origin_enc"),
    ]:
        le = LabelEncoder()
        src = df[col].fillna("UNKNOWN").astype(str) if col == "origin_state" else df[col].astype(str)
        df[key] = le.fit_transform(src)
        encoders[key] = le
    return df, encoders

df, encoders = encode_categoricals(df)


# =============================================================================
# STEP 4: TRAIN/TEST SPLIT  (before target encoding to prevent leakage)
# =============================================================================
TARGET = "arr_del15"

# NOTE: origin_enc intentionally excluded.
# route_delay_rate already contains the origin signal — keeping origin_delay_rate
# separately was creating negative interference (LR coefficient was -0.515).
# route captures the origin+dest pair which is strictly more informative.
FEATURES_BASE = [
    "month",
    "day_of_week",
    "is_weekend",
    "dep_hour",
    "dep_period",
    "distance",
    "dest_enc",
    "carrier_enc",
    "state_enc",
    "route_enc",
    "carrier_origin_enc",
]

model_df = df[FEATURES_BASE + [TARGET]].dropna()

X = model_df[FEATURES_BASE]
y = model_df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"Delayed in test set: {y_test.sum():,} ({y_test.mean():.1%})")


# =============================================================================
# STEP 4B: TARGET ENCODING
# =============================================================================

def target_encode(train_df, test_df, col, target_col, smoothing=20):
    """
    Smoothed target encoding.
    encoded = (n * mean_cat + smoothing * global_mean) / (n + smoothing)
    Higher smoothing = rare categories pulled harder toward global mean.
    """
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
               (stats["count"] + smoothing)
    return (
        train_df[col].map(smoothed).fillna(global_mean),
        test_df[col].map(smoothed).fillna(global_mean)
    )

train_with_target = X_train.copy()
train_with_target[TARGET] = y_train.values

# Smoothing values:
#   dest/carrier/state — smoothing=20 (enough data per category)
#   route/carrier_origin — smoothing=50 (many rare combinations; pull harder toward mean)
ENCODE_SETTINGS = {
    "dest_enc":           20,
    "carrier_enc":        20,
    "state_enc":          20,
    "route_enc":          50,   # tuned up from 20
    "carrier_origin_enc": 50,   # tuned up from 20
}

for col, smoothing in ENCODE_SETTINGS.items():
    new_col = col.replace("_enc", "_delay_rate")
    X_train[new_col], X_test[new_col] = target_encode(
        train_with_target, X_test, col, TARGET, smoothing=smoothing
    )

X_train = X_train.drop(columns=list(ENCODE_SETTINGS.keys()))
X_test  = X_test.drop(columns=list(ENCODE_SETTINGS.keys()))

FEATURES_FINAL = list(X_train.columns)
print(f"\nFinal features after target encoding ({len(FEATURES_FINAL)} total):")
print(FEATURES_FINAL)


# =============================================================================
# STEP 5A: LOGISTIC REGRESSION (Baseline)
# =============================================================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

lr_metrics = {
    "Accuracy":  accuracy_score(y_test, lr_preds),
    "Precision": precision_score(y_test, lr_preds),
    "Recall":    recall_score(y_test, lr_preds),
    "F1 Score":  f1_score(y_test, lr_preds),
    "AUC-ROC":   roc_auc_score(y_test, lr_proba),
}
print(pd.Series(lr_metrics).round(4).to_string())
print("\n", classification_report(y_test, lr_preds, target_names=["On Time (0)", "Delayed (1)"]))

coef_df = pd.DataFrame({
    "Feature": FEATURES_FINAL, "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", ascending=False)
print("\nCoefficients (positive = pushes toward predicting delay):")
print(coef_df.to_string(index=False))


# =============================================================================
# STEP 5B: RANDOM FOREST (tuned)
# =============================================================================
print("\n" + "="*50)
print("RANDOM FOREST (tuned)")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_metrics = {
    "Accuracy":  accuracy_score(y_test, rf_preds),
    "Precision": precision_score(y_test, rf_preds),
    "Recall":    recall_score(y_test, rf_preds),
    "F1 Score":  f1_score(y_test, rf_preds),
    "AUC-ROC":   roc_auc_score(y_test, rf_proba),
}
print(pd.Series(rf_metrics).round(4).to_string())
print("\n", classification_report(y_test, rf_preds, target_names=["On Time (0)", "Delayed (1)"]))


# =============================================================================
# STEP 5C: XGBOOST (ceiling raised — 3000 trees, patience=50)
# =============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*50)
    print("XGBOOST")
    print("="*50)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        n_estimators=3000,         # raised from 1000 — was still learning at 999
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,  # raised from 30 — more patience
        verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    print(f"Best iteration: {xgb_model.best_iteration}")

    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    xgb_metrics = {
        "Accuracy":  accuracy_score(y_test, xgb_preds),
        "Precision": precision_score(y_test, xgb_preds),
        "Recall":    recall_score(y_test, xgb_preds),
        "F1 Score":  f1_score(y_test, xgb_preds),
        "AUC-ROC":   roc_auc_score(y_test, xgb_proba),
    }
    print(pd.Series(xgb_metrics).round(4).to_string())
    print("\n", classification_report(y_test, xgb_preds, target_names=["On Time (0)", "Delayed (1)"]))
else:
    xgb_metrics = xgb_preds = xgb_proba = xgb_model = None


# =============================================================================
# STEP 5D: LIGHTGBM (fixed — is_unbalance=True, min_child_samples=50)
# Previous version collapsed at iteration 3 because scale_pos_weight
# combined with early stopping caused the model to predict all zeros.
# is_unbalance=True is LightGBM's native imbalance handler — more stable.
# =============================================================================
if LGBM_AVAILABLE:
    print("\n" + "="*50)
    print("LIGHTGBM (fixed)")
    print("="*50)

    lgbm_model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        is_unbalance=True,     # replaces scale_pos_weight — LightGBM's native flag
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,  # raised from 20 — stabilises early rounds
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

    lgbm_preds = lgbm_model.predict(X_test)
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]

    lgbm_metrics = {
        "Accuracy":  accuracy_score(y_test, lgbm_preds),
        "Precision": precision_score(y_test, lgbm_preds),
        "Recall":    recall_score(y_test, lgbm_preds),
        "F1 Score":  f1_score(y_test, lgbm_preds),
        "AUC-ROC":   roc_auc_score(y_test, lgbm_proba),
    }
    print(pd.Series(lgbm_metrics).round(4).to_string())
    print("\n", classification_report(y_test, lgbm_preds, target_names=["On Time (0)", "Delayed (1)"]))
else:
    lgbm_metrics = lgbm_preds = lgbm_proba = lgbm_model = None


# =============================================================================
# STEP 6: VISUALIZATIONS (1M exploration run)
# =============================================================================

model_names        = ["Logistic Regression", "Random Forest"]
model_preds        = [lr_preds,  rf_preds]
model_probas       = [lr_proba,  rf_proba]
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

n_models = len(model_names)

# --- 6A: Confusion Matrices ---
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]
for ax, preds, title in zip(axes, model_preds, model_names):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On Time", "Delayed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")
plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6B: ROC Curves ---
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
fig, ax = plt.subplots(figsize=(8, 6))
for name, proba, metrics, color in zip(model_names, model_probas, model_metrics_list, colors):
    RocCurveDisplay.from_predictions(
        y_test, proba,
        name=f"{name} (AUC={metrics['AUC-ROC']:.3f})",
        ax=ax, color=color
    )
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax.set_title("ROC Curves — Flight Delay Classification", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6C: Feature Importance (Random Forest) ---
imp_rf = pd.Series(rf.feature_importances_, index=FEATURES_FINAL).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
imp_rf.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/feature_importance_rf.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6D: Feature Importance (XGBoost) ---
if XGBOOST_AVAILABLE:
    imp_xgb = pd.Series(
        xgb_model.feature_importances_, index=FEATURES_FINAL
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    imp_xgb.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_xgb.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- 6E: Feature Importance (LightGBM) ---
if LGBM_AVAILABLE:
    imp_lgbm = pd.Series(
        lgbm_model.feature_importances_, index=FEATURES_FINAL
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    imp_lgbm.plot(kind="barh", ax=ax, color="mediumseagreen")
    ax.set_title("LightGBM — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_lgbm.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- 6F: Metrics Comparison ---
comparison = pd.DataFrame(
    {name: m for name, m in zip(model_names, model_metrics_list)}
).T
fig, ax = plt.subplots(figsize=(12, 4))
comparison.plot(kind="bar", ax=ax, ylim=(0, 1), colormap="Set2", edgecolor="black")
ax.set_title("Model Performance Comparison (1M sample)", fontsize=13, fontweight="bold")
ax.set_xticklabels(comparison.index, rotation=0)
ax.legend(loc="lower right")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== Exploration Run Results (1M rows) ===")
print(comparison.round(4).to_string())


# =============================================================================
# STEP 7: FULL 6.4M ROW FINAL RUN (XGBoost + LightGBM only)
# Random Forest is too slow on 6.4M rows. LR and RF numbers from above
# are already reportable — this step improves the two gradient boosted models.
#
# NOTE: This will take ~15-30 minutes depending on your machine.
#       Comment out this entire block if you just need the quick results.
# =============================================================================
print("\n" + "="*60)
print("STEP 7: FULL DATA RUN (6.4M rows) — XGBoost + LightGBM")
print("This may take 15-30 minutes. Comment out if not needed.")
print("="*60)

# Re-engineer features on full dataset using same function
df_full = engineer_features(df_full)

# Re-encode categoricals (fit on full data)
df_full, _ = encode_categoricals(df_full)

full_model_df = df_full[FEATURES_BASE + [TARGET]].dropna()
X_full = full_model_df[FEATURES_BASE]
y_full = full_model_df[TARGET].astype(int)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
print(f"\nFull run — Train: {len(X_train_f):,} rows  |  Test: {len(X_test_f):,} rows")

# Target encode using same settings — fit on full training set
train_full_with_target = X_train_f.copy()
train_full_with_target[TARGET] = y_train_f.values

for col, smoothing in ENCODE_SETTINGS.items():
    new_col = col.replace("_enc", "_delay_rate")
    X_train_f[new_col], X_test_f[new_col] = target_encode(
        train_full_with_target, X_test_f, col, TARGET, smoothing=smoothing
    )

X_train_f = X_train_f.drop(columns=list(ENCODE_SETTINGS.keys()))
X_test_f  = X_test_f.drop(columns=list(ENCODE_SETTINGS.keys()))

# --- Full XGBoost ---
if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost on full 6.4M rows...")
    scale_pw_f = (y_train_f == 0).sum() / (y_train_f == 1).sum()

    xgb_full = XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pw_f,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_full.fit(
        X_train_f, y_train_f,
        eval_set=[(X_test_f, y_test_f)],
        verbose=100
    )
    print(f"Best iteration: {xgb_full.best_iteration}")

    xgb_full_proba = xgb_full.predict_proba(X_test_f)[:, 1]
    xgb_full_preds = xgb_full.predict(X_test_f)
    print(f"\nXGBoost FULL RUN AUC-ROC: {roc_auc_score(y_test_f, xgb_full_proba):.4f}")
    print(f"XGBoost FULL RUN F1:      {f1_score(y_test_f, xgb_full_preds):.4f}")

# --- Full LightGBM ---
if LGBM_AVAILABLE:
    print("\nTraining LightGBM on full 6.4M rows...")

    lgbm_full = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        random_state=42,
        n_jobs=-1,
        verbose=-1
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
    lgbm_full_preds = lgbm_full.predict(X_test_f)
    print(f"\nLightGBM FULL RUN AUC-ROC: {roc_auc_score(y_test_f, lgbm_full_proba):.4f}")
    print(f"LightGBM FULL RUN F1:      {f1_score(y_test_f, lgbm_full_preds):.4f}")

# --- Full run ROC curve ---
if XGBOOST_AVAILABLE and LGBM_AVAILABLE:
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test_f, xgb_full_proba,
        name=f"XGBoost full 6.4M (AUC={roc_auc_score(y_test_f, xgb_full_proba):.3f})",
        ax=ax, color="tab:green"
    )
    RocCurveDisplay.from_predictions(
        y_test_f, lgbm_full_proba,
        name=f"LightGBM full 6.4M (AUC={roc_auc_score(y_test_f, lgbm_full_proba):.3f})",
        ax=ax, color="tab:red"
    )
    # also overlay the 1M versions for comparison
    if XGBOOST_AVAILABLE:
        RocCurveDisplay.from_predictions(
            y_test, xgb_proba,
            name=f"XGBoost 1M sample (AUC={xgb_metrics['AUC-ROC']:.3f})",
            ax=ax, color="tab:orange", linestyle="--"
        )
    if LGBM_AVAILABLE:
        RocCurveDisplay.from_predictions(
            y_test, lgbm_proba,
            name=f"LightGBM 1M sample (AUC={lgbm_metrics['AUC-ROC']:.3f})",
            ax=ax, color="tab:purple", linestyle="--"
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_title("ROC Curves — 1M Sample vs Full 6.4M", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("charts/roc_curves_full_run.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nAll charts saved.")


# =============================================================================
# NOTES
# =============================================================================
# 1. DB_PATH: update to your actual path.
#
# 2. dep_del15 excluded — pre-departure framing. See project plan Sec 5.2.
#
# 3. origin_enc excluded — route_delay_rate already contains the origin signal.
#    Including both caused negative interference (LR coef was -0.515).
#
# 4. Smoothing=50 for route and carrier_origin — many rare combinations
#    exist that need stronger pulling toward the global mean.
#
# 5. LightGBM fix: is_unbalance=True replaces scale_pos_weight.
#    The previous version collapsed at iteration 3 due to interaction
#    between scale_pos_weight and early stopping in early rounds.
#
# 6. XGBoost fix: n_estimators=3000 with early_stopping_rounds=50.
#    Previous version hit ceiling at iteration 999 and was still learning.
#
# 7. Step 7 (full 6.4M run) can be commented out to get quick results.
#    Expected time: 15-30 min depending on CPU. LightGBM will be ~3x faster.
#
# 8. Install: pip install xgboost lightgbm
#
# 9. Delay cause columns excluded — recorded after landing (data leakage).
# =============================================================================
