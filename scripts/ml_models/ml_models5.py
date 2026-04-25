# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15 (already a boolean in the flight table)
#
# CHANGES FROM PREVIOUS VERSION:
#   1. Added route_enc  — origin+dest pair target-encoded (biggest new signal)
#   2. Added carrier_origin_enc — carrier+origin interaction target-encoded
#   3. XGBoost retuned — n_estimators=1000, lr=0.05, early_stopping_rounds=30
#   4. Added LightGBM model — often beats XGBoost on large tabular data
#   5. Visualizations updated to handle 4 models
#
#   dep_del15 intentionally excluded — pre-departure prediction framing
#   (see project plan Section 5.2 and NOTE in FEATURES_BASE below)
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

# Sample to 1M rows
df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows")

print(f"\nClass balance (arr_del15):\n{df['arr_del15'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

# --- Date features ---
df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
df["month"]       = df["flight_date"].dt.month       # 1-12
df["day_of_week"] = df["flight_date"].dt.dayofweek   # 0=Mon, 6=Sun

# --- Weekend flag ---
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# --- Departure hour ---
df["dep_hour"] = pd.to_datetime(
    df["crs_dep_time"], format="%H:%M:%S", errors="coerce"
).dt.hour
# If above returns all NaT try:
# df["dep_hour"] = df["crs_dep_time"].astype(float).floordiv(100).astype(int)

# --- Departure period buckets ---
df["dep_period"] = pd.cut(
    df["dep_hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=[0, 1, 2, 3, 4]   # 0=red-eye 1=morning 2=afternoon 3=evening 4=night
).astype("Int64").fillna(2).astype(int)

# --- NEW: Route key (origin+dest pair) ---
# ATL->JFK has a completely different delay profile than ATL->CLT.
# Encoding the pair captures what neither airport alone can.
df["route"] = df["origin"].astype(str) + "_" + df["dest"].astype(str)

# --- NEW: Carrier+Origin interaction key ---
# Spirit at ORD behaves very differently from Delta at ORD.
df["carrier_origin"] = df["carrier"].astype(str) + "_" + df["origin"].astype(str)

print("\nSample of engineered features:")
print(df[["flight_date", "month", "day_of_week", "is_weekend",
          "dep_hour", "dep_period", "distance", "route", "arr_del15"]].head())


# =============================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =============================================================================

le_origin         = LabelEncoder()
le_dest           = LabelEncoder()
le_state          = LabelEncoder()
le_carrier        = LabelEncoder()
le_route          = LabelEncoder()
le_carrier_origin = LabelEncoder()

df["origin_enc"]         = le_origin.fit_transform(df["origin"].astype(str))
df["dest_enc"]           = le_dest.fit_transform(df["dest"].astype(str))
df["state_enc"]          = le_state.fit_transform(df["origin_state"].fillna("UNKNOWN").astype(str))
df["carrier_enc"]        = le_carrier.fit_transform(df["carrier"].astype(str))
df["route_enc"]          = le_route.fit_transform(df["route"].astype(str))
df["carrier_origin_enc"] = le_carrier_origin.fit_transform(df["carrier_origin"].astype(str))


# =============================================================================
# STEP 4: TRAIN/TEST SPLIT  (done BEFORE target encoding to prevent leakage)
# =============================================================================
TARGET = "arr_del15"

FEATURES_BASE = [
    # NOTE: dep_del15 intentionally excluded.
    # Pre-departure prediction — dep_del15 is unknown before the flight leaves.
    "month",
    "day_of_week",
    "is_weekend",
    "dep_hour",
    "dep_period",
    "distance",
    "origin_enc",
    "dest_enc",
    "carrier_enc",
    "state_enc",
    "route_enc",           # NEW: origin+dest pair
    "carrier_origin_enc",  # NEW: carrier+hub interaction
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
# Each categorical is replaced with its smoothed historical delay rate,
# computed on the training set only to prevent leakage.
# =============================================================================

def target_encode(train_df, test_df, col, target_col, smoothing=20):
    """
    Smoothed target encoding:
      encoded = (n * mean_cat + smoothing * global_mean) / (n + smoothing)
    Smoothing pulls rare categories toward global mean to prevent overfitting.
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

ENCODE_COLS = [
    "origin_enc",
    "dest_enc",
    "carrier_enc",
    "state_enc",
    "route_enc",           # NEW
    "carrier_origin_enc",  # NEW
]

for col in ENCODE_COLS:
    new_col = col.replace("_enc", "_delay_rate")
    X_train[new_col], X_test[new_col] = target_encode(
        train_with_target, X_test, col, TARGET
    )

# Drop raw label-encoded integers — delay rates carry the actual signal
X_train = X_train.drop(columns=ENCODE_COLS)
X_test  = X_test.drop(columns=ENCODE_COLS)

FEATURES_FINAL = list(X_train.columns)
print(f"\nFinal features after target encoding ({len(FEATURES_FINAL)} total):")
print(FEATURES_FINAL)


# =============================================================================
# STEP 5A: LOGISTIC REGRESSION (Baseline)
# =============================================================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
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
    "Feature": FEATURES_FINAL,
    "Coefficient": lr.coef_[0]
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
    n_estimators=200,
    max_depth=20,
    min_samples_split=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
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
# STEP 5C: XGBOOST (retuned with early stopping)
# =============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*50)
    print("XGBOOST (retuned)")
    print("="*50)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        n_estimators=1000,         # high ceiling — early stopping finds the sweet spot
        max_depth=6,
        learning_rate=0.05,        # slower learning generalises better than 0.1
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,       # prevents splits on tiny leaf groups
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=30,  # stops when val AUC plateaus for 30 rounds
        verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100   # prints AUC every 100 rounds so you can watch progress
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
# STEP 5D: LIGHTGBM (new — fastest on large tabular data, often best AUC)
# =============================================================================
if LGBM_AVAILABLE:
    print("\n" + "="*50)
    print("LIGHTGBM")
    print("="*50)

    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()

    lgbm_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,         # 2^max_depth - 1 is a good starting rule
        scale_pos_weight=scale_pw,
        subsample=0.8,
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
            lgb.early_stopping(stopping_rounds=30, verbose=True),
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
# STEP 6: VISUALIZATIONS
# =============================================================================

# Collect all active models dynamically
model_names        = ["Logistic Regression", "Random Forest"]
model_preds        = [lr_preds,  rf_preds]
model_probas       = [lr_proba,  rf_proba]
model_metrics_list = [lr_metrics, rf_metrics]
model_importances  = [
    pd.Series(rf.feature_importances_, index=FEATURES_FINAL),
    None  # LR has coefficients not importances — handled separately
]

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

# --- 6B: ROC Curves (all models on one chart) ---
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

# --- 6F: Final Metrics Comparison Bar Chart ---
comparison = pd.DataFrame(
    {name: m for name, m in zip(model_names, model_metrics_list)}
).T

fig, ax = plt.subplots(figsize=(12, 4))
comparison.plot(kind="bar", ax=ax, ylim=(0, 1), colormap="Set2", edgecolor="black")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.set_xticklabels(comparison.index, rotation=0)
ax.legend(loc="lower right")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== Final Model Comparison ===")
print(comparison.round(4).to_string())
print("\nCharts saved:")
print("  confusion_matrices.png, roc_curves.png, model_comparison.png")
print("  feature_importance_rf.png, feature_importance_xgb.png, feature_importance_lgbm.png")


# =============================================================================
# NOTES
# =============================================================================
# 1. DB_PATH: update to your actual path.
#
# 2. dep_del15 intentionally excluded — pre-departure prediction framing.
#    Including it inflates AUC but makes the model useless before departure.
#    See project plan Section 5.2.
#
# 3. dep_hour parsing: if pd.to_datetime returns all NaT try:
#    df["dep_hour"] = df["crs_dep_time"].astype(float).floordiv(100).astype(int)
#
# 4. Target encoding: computed on training set only, applied to test set.
#    Smoothing=20 pulls rare routes/carrier-hub combos toward global mean.
#
# 5. Install: pip install xgboost lightgbm
#
# 6. Early stopping: both XGBoost and LightGBM stop automatically when
#    validation AUC stops improving for 30 rounds. Console prints progress
#    every 100 rounds. Best iteration is printed after training completes.
#
# 7. Delay cause columns (carrier_delay, weather_delay, etc.) excluded —
#    recorded after landing, using them is data leakage.
#
# 8. Honest AUC ceiling: ~0.76-0.80 is realistic for pre-departure features.
#    NOAA weather data at origin airports would be the next meaningful step.
# =============================================================================
