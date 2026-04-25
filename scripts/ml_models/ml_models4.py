# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15 (already a boolean in the flight table)
#
# CHANGES FROM CHECKPOINT 4:
#   1. Added dest_enc to features
#   2. Added is_weekend + dep_period interaction features
#   3. Added target encoding for origin/dest/carrier (replaces raw LabelEncoder)
#   4. Tuned Random Forest (200 trees, depth 20, all cores)
#   5. Added XGBoost model (typically best AUC on tabular data)
#   6. Added final 3-model comparison chart
#
#   dep_del15 intentionally excluded — pre-departure prediction framing
#   (see project plan Section 5.2 and NOTE in FEATURES_BASE below)
# =============================================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    print("Continuing with Logistic Regression + Random Forest only.\n")
    XGBOOST_AVAILABLE = False

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

# Downcast float64 → float32 to halve RAM usage
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
df["month"]       = df["flight_date"].dt.month        # 1–12
df["day_of_week"] = df["flight_date"].dt.dayofweek    # 0=Mon, 6=Sun

# --- NEW: Weekend flag ---
# Weekends show distinct delay patterns vs. weekdays
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# --- Departure hour ---
df["dep_hour"] = pd.to_datetime(
    df["crs_dep_time"], format="%H:%M:%S", errors="coerce"
).dt.hour

# If the above returns all NaT, SQLite may store time as integer HHMM — try:
# df["dep_hour"] = (df["crs_dep_time"].astype(float) // 100).astype(int)

# --- NEW: Departure period buckets ---
# Converts raw 0–23 hour into 5 meaningful bins the model can use more easily
df["dep_period"] = pd.cut(
    df["dep_hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=[0, 1, 2, 3, 4]          # 0=red-eye, 1=morning, 2=afternoon, 3=evening, 4=night
).astype("Int64").fillna(2).astype(int)

print("\nSample of engineered features:")
print(df[["flight_date", "month", "day_of_week", "is_weekend",
          "dep_hour", "dep_period", "distance", "arr_del15"]].head())


# =============================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =============================================================================

# --- Label-encode to get integer keys (used for target encoding below) ---
le_origin  = LabelEncoder()
le_dest    = LabelEncoder()
le_state   = LabelEncoder()
le_carrier = LabelEncoder()

df["origin_enc"]  = le_origin.fit_transform(df["origin"].astype(str))
df["dest_enc"]    = le_dest.fit_transform(df["dest"].astype(str))
df["state_enc"]   = le_state.fit_transform(df["origin_state"].fillna("UNKNOWN").astype(str))
df["carrier_enc"] = le_carrier.fit_transform(df["carrier"].astype(str))


# =============================================================================
# STEP 4: TRAIN/TEST SPLIT (done BEFORE target encoding to prevent leakage)
# =============================================================================
TARGET = "arr_del15"

FEATURES_BASE = [
    # NOTE: dep_del15 is intentionally excluded.
    # This model predicts arrival delay BEFORE departure — dep_del15 is unknown
    # at prediction time and including it would make the model useless in practice.
    "month",
    "day_of_week",
    "is_weekend",     # new
    "dep_hour",
    "dep_period",     # new
    "distance",
    "origin_enc",
    "dest_enc",       # added
    "carrier_enc",
    "state_enc",
]

model_df = df[FEATURES_BASE + [TARGET]].dropna()

X = model_df[FEATURES_BASE]
y = model_df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nFeatures used: {FEATURES_BASE}")
print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"Delayed in test set: {y_test.sum():,} ({y_test.mean():.1%})")


# =============================================================================
# STEP 4B: TARGET ENCODING
# Replace raw label-encoded integers with each airport/carrier's historical
# delay rate — computed on train set only to prevent leakage.
# =============================================================================

def target_encode(train_df, test_df, col, target_col, smoothing=20):
    """
    Smoothed target encoding:
      encoded = (n_cat * mean_cat + smoothing * global_mean) / (n_cat + smoothing)
    Smoothing pulls rare categories toward the global mean to avoid overfitting.
    """
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
               (stats["count"] + smoothing)
    train_encoded = train_df[col].map(smoothed).fillna(global_mean)
    test_encoded  = test_df[col].map(smoothed).fillna(global_mean)
    return train_encoded, test_encoded

# Build combined train+target df for encoding
train_with_target = X_train.copy()
train_with_target[TARGET] = y_train.values

for col in ["origin_enc", "dest_enc", "carrier_enc", "state_enc"]:
    new_col = col.replace("_enc", "_delay_rate")
    X_train[new_col], X_test[new_col] = target_encode(
        train_with_target, X_test, col, TARGET
    )

# Drop the raw label-encoded columns — delay rates are strictly better
X_train = X_train.drop(columns=["origin_enc", "dest_enc", "carrier_enc", "state_enc"])
X_test  = X_test.drop(columns=["origin_enc", "dest_enc", "carrier_enc", "state_enc"])

FEATURES_FINAL = list(X_train.columns)
print(f"\nFinal features after target encoding: {FEATURES_FINAL}")


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
    n_estimators=200,      # was 50 — more trees = more stable predictions
    max_depth=20,          # was 10 — allows the model to capture deeper patterns
    min_samples_split=20,  # was 50 — allows finer splits on large node
    class_weight="balanced",
    random_state=42,
    n_jobs=-1              # was 2 — use all available cores
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
# STEP 5C: XGBOOST (new — typically best AUC on tabular data)
# =============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*50)
    print("XGBOOST")
    print("="*50)

    # scale_pos_weight handles class imbalance the same way class_weight="balanced" does
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f} (auto-computed from training set)")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        verbosity=0
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

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
    xgb_metrics = None
    xgb_preds  = None
    xgb_proba  = None


# =============================================================================
# STEP 6: VISUALIZATIONS
# =============================================================================

# Collect active models
model_names  = ["Logistic Regression", "Random Forest"]
model_preds  = [lr_preds,  rf_preds]
model_probas = [lr_proba,  rf_proba]
model_metrics_list = [lr_metrics, rf_metrics]

if XGBOOST_AVAILABLE:
    model_names.append("XGBoost")
    model_preds.append(xgb_preds)
    model_probas.append(xgb_proba)
    model_metrics_list.append(xgb_metrics)

n_models = len(model_names)

# --- 6A: Confusion Matrices ---
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]
for ax, preds, title in zip(axes, model_preds, model_names):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On Time", "Delayed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6B: ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["tab:blue", "tab:orange", "tab:green"]
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
importances = pd.Series(rf.feature_importances_, index=FEATURES_FINAL).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/feature_importance_rf.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6D: Feature Importance (XGBoost) ---
if XGBOOST_AVAILABLE:
    xgb_imp = pd.Series(
        xgb.feature_importances_, index=FEATURES_FINAL
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    xgb_imp.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_xgb.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- 6E: Metrics Comparison Bar Chart ---
comparison = pd.DataFrame(
    {name: m for name, m in zip(model_names, model_metrics_list)}
).T

fig, ax = plt.subplots(figsize=(10, 4))
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
print("\nCharts saved: confusion_matrices.png, roc_curves.png,")
print("              feature_importance_rf.png, feature_importance_xgb.png, model_comparison.png")


# =============================================================================
# NOTES
# =============================================================================
# 1. DB_PATH: update to your actual path.
#
# 2. dep_del15 is intentionally excluded from features. This model is framed
#    as a PRE-DEPARTURE predictor — a passenger or airline querying it before
#    the flight leaves does not yet know whether departure was delayed.
#    Including dep_del15 would inflate AUC significantly but make the model
#    useless in any real-world scenario. See project plan Section 5.2.
#
# 3. dep_hour parsing: SQLite stores TIME as text "HH:MM:SS". If pd.to_datetime
#    fails, check df["crs_dep_time"].head() and adjust the format string.
#    If stored as integer HHMM, use: df["dep_hour"] = df["crs_dep_time"] // 100
#
# 4. Target encoding: computed on train set only and applied to test set.
#    This prevents leakage while giving the model a meaningful continuous
#    signal (historical delay rate) instead of an arbitrary integer label.
#
# 5. XGBoost install: pip install xgboost
#    On a 1M-row dataset, XGBoost training should take ~2–5 minutes.
#
# 6. The delay cause columns (carrier_delay, weather_delay, etc.) are still
#    intentionally excluded — they are recorded AFTER landing (data leakage).
#
# 7. To push AUC even further: consider adding a route_delay_rate feature
#    (target encode the origin+dest pair) and experimenting with
#    n_estimators=500 in XGBoost.
# =============================================================================
